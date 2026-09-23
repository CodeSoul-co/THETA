const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('thetaDesktop', {
  read: () => ipcRenderer.invoke('settings:read'),
  catalog: () => ipcRenderer.invoke('settings:catalog'),
  saveInference: value => ipcRenderer.invoke('settings:save-inference', value),
  saveEmbedding: value => ipcRenderer.invoke('settings:save-embedding', value),
  selectModel: () => ipcRenderer.invoke('settings:select-model'),
  openModels: kind => ipcRenderer.invoke('settings:open-models', kind),
  openData: () => ipcRenderer.invoke('settings:open-data'),
  onOpenSettings: callback => {
    const listener = () => callback();
    ipcRenderer.on('desktop:open-settings', listener);
    return () => ipcRenderer.removeListener('desktop:open-settings', listener);
  },
});
