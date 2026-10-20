import * as vscode from 'vscode';
import { ChatPanel } from './chatPanel';

export function activate(context: vscode.ExtensionContext): void {
  const outputChannel = vscode.window.createOutputChannel('Forge');
  context.subscriptions.push(outputChannel);

  context.subscriptions.push(
    vscode.commands.registerCommand('forge.openChat', () => {
      ChatPanel.createOrShow(context.extensionUri, outputChannel);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('forge.newSession', () => {
      ChatPanel.current?.dispose();
      ChatPanel.createOrShow(context.extensionUri, outputChannel);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('forge.stopSession', () => {
      ChatPanel.current?.cancelCurrentTurn();
    }),
  );
}

export function deactivate(): void {
  ChatPanel.current?.dispose();
}
