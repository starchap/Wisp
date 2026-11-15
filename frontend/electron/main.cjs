const { app, BrowserWindow, dialog, ipcMain, session, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow = null;
let pythonProcess = null;

ipcMain.handle('wisp:selectDirectory', async (_event, defaultPath) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select WispAI Storage Folder (Models & Recordings)',
    defaultPath: defaultPath || undefined,
    properties: ['openDirectory', 'createDirectory'],
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

function stopPythonBackend() {
  if (!pythonProcess) return;
  const proc = pythonProcess;
  pythonProcess = null;
  try {
    if (process.platform === 'win32' && proc.pid) {
      spawn('taskkill', ['/pid', String(proc.pid), '/f', '/t'], {
        windowsHide: true,
        stdio: 'ignore',
      });
    } else {
      proc.kill('SIGTERM');
    }
  } catch (_) {}
}

function startPythonBackend() {
  const isWin = process.platform === 'win32';

  // 1. In packaged builds, prefer the bundled PyInstaller executable if present
  if (app.isPackaged) {
    const exeName = isWin ? 'wisp-backend.exe' : 'wisp-backend';
    const bundledExe = path.join(
      process.resourcesPath,
      'backend-dist',
      'wisp-backend',
      exeName
    );
    if (fs.existsSync(bundledExe)) {
      try {
        pythonProcess = spawn(bundledExe, [], {
          cwd: path.dirname(bundledExe),
          stdio: ['ignore', 'ignore', 'ignore'],
          windowsHide: true,
        });
        return;
      } catch (e) {
        console.error('[Electron] Failed to launch bundled backend binary:', e);
      }
    }
  }

  // 2. Development / Script fallback
  const backendDir = app.isPackaged
    ? path.join(process.resourcesPath, 'backend')
    : path.resolve(__dirname, '..', '..', 'backend');
  const mainScript = path.join(backendDir, 'main.py');

  const cmd = isWin ? 'py' : 'python3';
  const args = isWin ? ['-3.10', mainScript] : [mainScript];

  try {
    pythonProcess = spawn(cmd, args, {
      cwd: backendDir,
      stdio: 'ignore',
      windowsHide: true,
    });
    pythonProcess.on('error', (err) => {
      console.log('[Electron] Fallback to python executable:', err.message);
      pythonProcess = spawn('python', [mainScript], {
        cwd: backendDir,
        stdio: 'ignore',
        windowsHide: true,
      });
    });
  } catch (e) {
    console.error('[Electron] Could not spawn Python backend:', e);
  }
}

function createWindow() {
  const iconFile =
    process.platform === 'win32'
      ? path.join(__dirname, '..', 'public', 'wisp-icon.ico')
      : path.join(__dirname, '..', 'public', 'wisp-icon.png');

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1080,
    minHeight: 700,
    title: 'WispAI',
    icon: iconFile,
    backgroundColor: '#020617',
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
  });

  const distIndex = path.join(__dirname, '..', 'dist', 'index.html');
  const devUrl = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

  if (fs.existsSync(distIndex) && !process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadFile(distIndex);
  } else {
    mainWindow.loadURL(devUrl).catch(() => {
      if (fs.existsSync(distIndex)) {
        mainWindow.loadFile(distIndex);
      }
    });
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
    mainWindow.setAlwaysOnTop(true);
    setTimeout(() => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.setAlwaysOnTop(false);
      }
    }, 1200);
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

const gotSingleLock = app.requestSingleInstanceLock();
if (!gotSingleLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    // Technical Privacy Guarantee: Block any outbound network requests from the UI renderer
    // except local loopback (127.0.0.1 / localhost) and local file/blob/data protocols.
    session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
      const url = details.url || '';
      const isAllowedLocal =
        url.startsWith('file://') ||
        url.startsWith('data:') ||
        url.startsWith('blob:') ||
        url.startsWith('devtools://') ||
        url.startsWith('chrome-extension://') ||
        url.startsWith('http://127.0.0.1:') ||
        url.startsWith('ws://127.0.0.1:') ||
        url.startsWith('http://localhost:') ||
        url.startsWith('ws://localhost:');
      callback({ cancel: !isAllowedLocal });
    });

    if (process.env.SKIP_BACKEND_SPAWN !== '1') {
      startPythonBackend();
    }
    createWindow();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });
}

app.on('before-quit', () => {
  stopPythonBackend();
});

app.on('window-all-closed', () => {
  stopPythonBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
