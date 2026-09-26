const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('thetaDesktop', {
  read: () => ipcRenderer.invoke('settings:read'),
  catalog: () => ipcRenderer.invoke('settings:catalog'),
  saveInference: value => ipcRenderer.invoke('settings:save-inference', value),
  saveEmbedding: value => ipcRenderer.invoke('settings:save-embedding', value),
  selectModel: () => ipcRenderer.invoke('settings:select-model'),
  openModels: kind => ipcRenderer.invoke('settings:open-models', kind),
  openData: () => ipcRenderer.invoke('settings:open-data'),
  updates: {
    state: () => ipcRenderer.invoke('updates:state'),
    check: () => ipcRenderer.invoke('updates:check'),
    download: () => ipcRenderer.invoke('updates:download'),
    install: () => ipcRenderer.invoke('updates:install'),
    configure: automatic => ipcRenderer.invoke('updates:configure', automatic),
    subscribe: callback => {
      const listener = (_event, state) => callback(state);
      ipcRenderer.on('desktop:update-state', listener);
      return () => ipcRenderer.removeListener('desktop:update-state', listener);
    },
  },
  onOpenUpdates: callback => {
    const listener = () => callback();
    ipcRenderer.on('desktop:open-updates', listener);
    return () => ipcRenderer.removeListener('desktop:open-updates', listener);
  },
  onOpenSettings: callback => {
    const listener = () => callback();
    ipcRenderer.on('desktop:open-settings', listener);
    return () => ipcRenderer.removeListener('desktop:open-settings', listener);
  },
});
