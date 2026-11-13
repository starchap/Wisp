const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow = null;
let pythonProcess = null;

ipcMain.handle('wisp:selectDirectory', async (_event, defaultPath) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select Wisp Storage Folder (Models & Recordings)',
    defaultPath: defaultPath || undefined,
    properties: ['openDirectory', 'createDirectory'],
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

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
          stdio: 'inherit',
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
      stdio: 'inherit',
      windowsHide: true,
    });
    pythonProcess.on('error', (err) => {
      console.log('[Electron] Fallback to python executable:', err.message);
      pythonProcess = spawn('python', [mainScript], {
        cwd: backendDir,
        stdio: 'inherit',
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
    title: 'Wisp',
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

app.whenReady().then(() => {
  if (process.env.SKIP_BACKEND_SPAWN !== '1') {
    startPythonBackend();
  }
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    try {
      pythonProcess.kill();
    } catch (_) {}
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
