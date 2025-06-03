import * as vscode from 'vscode';
import { exec } from 'child_process';

export function activate(context: vscode.ExtensionContext): void {
    const command = vscode.commands.registerCommand('skypilot.showQueue', () => {
        const panel = vscode.window.createWebviewPanel(
            'skypilotQueue',
            'Skypilot Queue',
            vscode.ViewColumn.One,
            { enableScripts: true }
        );

        const update = () => {
            exec('sky status', (err, stdout, stderr) => {
                if (err) {
                    panel.webview.html = `<pre>${stderr}</pre>`;
                    return;
                }
                panel.webview.html = `<pre>${stdout}</pre>`;
            });
        };

        update();
        const timer = setInterval(update, 10000);
        panel.onDidDispose(() => clearInterval(timer));
    });

    context.subscriptions.push(command);
}

export function deactivate(): void {}
