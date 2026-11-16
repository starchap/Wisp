const { app, BrowserWindow, dialog, ipcMain, session, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, spawnSync } = require('child_process');

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

function startScriptFallback(envVars) {
  const isWin = process.platform === 'win32';
  const backendDir = app.isPackaged
    ? path.join(process.resourcesPath, 'backend')
    : path.resolve(__dirname, '..', '..', 'backend');
  const mainScript = path.join(backendDir, 'main.py');

  // Check if a signed embedded python.exe is bundled in backend-dist/python
  if (app.isPackaged && isWin) {
    const signedBundledPy = path.join(
      process.resourcesPath,
      'backend-dist',
      'python',
      'python.exe'
    );
    if (fs.existsSync(signedBundledPy)) {
      try {
        const proc = spawn(signedBundledPy, [mainScript], {
          cwd: backendDir,
          env: envVars,
          stdio: 'ignore',
          windowsHide: true,
        });
        pythonProcess = proc;
        proc.on('error', () => {
          spawnSystemPython(backendDir, mainScript, envVars);
        });
        return;
      } catch (_) {}
    }
  }

  spawnSystemPython(backendDir, mainScript, envVars);
}

function spawnSystemPython(backendDir, mainScript, envVars) {
  const isWin = process.platform === 'win32';
  const cmd = isWin ? 'py' : 'python3';
  const args = isWin ? ['-3.10', mainScript] : [mainScript];

  try {
    const proc = spawn(cmd, args, {
      cwd: backendDir,
      env: envVars,
      stdio: 'ignore',
      windowsHide: true,
    });
    pythonProcess = proc;
    proc.on('error', (err) => {
      console.log('[Electron] Fallback to python executable:', err.message);
      try {
        const fallbackProc = spawn('python', [mainScript], {
          cwd: backendDir,
          env: envVars,
          stdio: 'ignore',
          windowsHide: true,
        });
        pythonProcess = fallbackProc;
        fallbackProc.on('error', () => {});
      } catch (_) {}
    });
  } catch (e) {
    console.error('[Electron] Could not spawn Python backend:', e);
  }
}

function startPythonBackend() {
  const isWin = process.platform === 'win32';
  const backendDir = app.isPackaged
    ? path.join(process.resourcesPath, 'backend')
    : path.resolve(__dirname, '..', '..', 'backend');
  const mainScript = path.join(backendDir, 'main.py');

  const envVars = {
    ...process.env,
    WISP_PACKAGED: app.isPackaged ? '1' : '0',
    PYTHONUTF8: '1',
    PYTHONDONTWRITEBYTECODE: '1',
  };

  // 1. On Windows packaged builds, prefer the DigiCert-signed embedded Python runtime
  // (signed by Python Software Foundation so Windows Smart App Control never blocks it).
  if (app.isPackaged && isWin) {
    const signedPyDir = path.join(process.resourcesPath, 'backend-dist', 'python');
    const signedPyExe = path.join(signedPyDir, 'python.exe');
    if (fs.existsSync(signedPyExe)) {
      try {
        const proc = spawn(signedPyExe, ['-E', '-s', mainScript], {
          cwd: backendDir,
          env: {
            ...envVars,
            PYTHONHOME: signedPyDir,
            PYTHONPATH: [
              backendDir,
              path.join(signedPyDir, 'Lib'),
              path.join(signedPyDir, 'Lib', 'site-packages'),
              path.join(signedPyDir, 'DLLs'),
            ].join(path.delimiter),
          },
          stdio: ['ignore', 'ignore', 'ignore'],
          windowsHide: true,
        });
        pythonProcess = proc;
        proc.on('error', (err) => {
          console.warn('[Electron] Signed embedded Python error, falling back:', err.message);
          spawnSystemPython(backendDir, mainScript, envVars);
        });
        return;
      } catch (e) {
        console.error('[Electron] Failed to launch signed embedded Python:', e);
      }
    }
  }

  // 2. In packaged builds (macOS / Linux / fallback), use the bundled PyInstaller executable
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
        const proc = spawn(bundledExe, [], {
          cwd: path.dirname(bundledExe),
          env: envVars,
          stdio: ['ignore', 'ignore', 'ignore'],
          windowsHide: true,
        });
        pythonProcess = proc;
        let fellBack = false;
        const triggerFallback = () => {
          if (fellBack) return;
          fellBack = true;
          startScriptFallback(envVars);
        };
        proc.on('error', (err) => {
          console.warn('[Electron] Bundled backend binary blocked or failed, falling back:', err.message);
          triggerFallback();
        });
        proc.on('exit', (code) => {
          if (code !== null && code !== 0) {
            console.warn('[Electron] Bundled backend exited early with code', code, '- falling back');
            triggerFallback();
          }
        });
        return;
      } catch (e) {
        console.error('[Electron] Failed to launch bundled backend binary:', e);
      }
    }
  }

  // 3. Development / Script fallback
  startScriptFallback(envVars);
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
