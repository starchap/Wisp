const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('wispElectron', {
  selectDirectory: (defaultPath) =>
    ipcRenderer.invoke('wisp:selectDirectory', defaultPath),
  getPathForFile: (file) => {
    try {
      if (webUtils && typeof webUtils.getPathForFile === 'function') {
        return webUtils.getPathForFile(file) || '';
      }
      return (file && file.path) || '';
    } catch (_) {
      return (file && file.path) || '';
    }
  },
});
