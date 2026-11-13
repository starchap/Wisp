const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('wispElectron', {
  selectDirectory: (defaultPath) =>
    ipcRenderer.invoke('wisp:selectDirectory', defaultPath),
});
