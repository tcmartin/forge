use super::*;
use agent_client_protocol_schema as acp;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Semaphore;

fn request(id: i64) -> Inbound {
    Inbound::Request {
        id: acp::RequestId::Number(id),
        method: "fixture".to_string(),
        params: None,
    }
}

#[tokio::test]
async fn retained_budget_lives_until_message_is_released() {
    let connection = Connection::new(tokio::io::sink());
    let charge = 64 * 1024 * 1024 + 1;
    let first = connection.admit_inbound(request(1), charge).unwrap();
    assert!(connection.admit_inbound(request(2), charge).is_err());
    drop(first);
    assert!(connection.admit_inbound(request(2), charge).is_ok());
}

#[tokio::test]
async fn handler_cap_rejects_excess_and_keeps_cancellation_responsive() {
    let (client, writer) = tokio::io::duplex(8192);
    let connection = Connection::new(writer);
    let (tx, rx) = mpsc::channel(MAX_INBOUND_QUEUE);
    let gates = Arc::new(Semaphore::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let (started, mut started_rx) = mpsc::channel(MAX_REQUEST_HANDLERS);
    let (cancelled, mut cancelled_rx) = mpsc::channel(1);
    let handler_gates = gates.clone();
    let handler_active = active.clone();
    let handler_peak = peak.clone();
    let dispatcher = tokio::spawn(dispatch_loop(connection.clone(), rx, move |message| {
        let gates = handler_gates.clone();
        let active = handler_active.clone();
        let peak = handler_peak.clone();
        let started = started.clone();
        let cancelled = cancelled.clone();
        async move {
            match &message.message {
                Inbound::Request { .. } => {
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                    started.send(()).await.unwrap();
                    gates.acquire().await.unwrap().forget();
                    active.fetch_sub(1, Ordering::SeqCst);
                }
                Inbound::Notification { .. } => cancelled.send(()).await.unwrap(),
            }
            drop(message);
        }
    }));
    for id in 0..MAX_REQUEST_HANDLERS {
        tx.send(connection.admit_inbound(request(id as i64), 1024).unwrap())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), started_rx.recv())
            .await
            .unwrap()
            .unwrap();
    }
    tx.send(connection.admit_inbound(request(99), 1024).unwrap())
        .await
        .unwrap();
    let mut reader = BufReader::new(client);
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(1), reader.read_line(&mut line))
        .await
        .unwrap()
        .unwrap();
    let response: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(response["id"], 99);
    assert_eq!(
        response["error"]["data"]["reason"],
        "ACP request handler limit reached"
    );
    tx.send(
        connection
            .admit_inbound(
                Inbound::Notification {
                    method: "session/cancel".to_string(),
                    params: None,
                },
                1024,
            )
            .unwrap(),
    )
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(1), cancelled_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(active.load(Ordering::SeqCst), MAX_REQUEST_HANDLERS);
    assert_eq!(peak.load(Ordering::SeqCst), MAX_REQUEST_HANDLERS);
    drop(tx);
    gates.add_permits(MAX_REQUEST_HANDLERS);
    tokio::time::timeout(Duration::from_secs(1), dispatcher)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(active.load(Ordering::SeqCst), 0);
    assert!(connection
        .admit_inbound(request(100), 128 * 1024 * 1024)
        .is_ok());
}
