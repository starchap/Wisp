const fs = require('fs');
const path = require('path');

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== 'win32') {
    return;
  }
  const appName = context.packager.appInfo.productFilename;
  const targetExe = path.join(context.appOutDir, `${appName}.exe`);
  const stockElectronExe = path.join(
    __dirname,
    '..',
    'node_modules',
    'electron',
    'dist',
    'electron.exe'
  );

  if (fs.existsSync(stockElectronExe) && fs.existsSync(targetExe)) {
    console.log(
      `[afterPack] Restoring unmodified official electron.exe binary to ${targetExe} so Windows Smart App Control recognizes its trusted SHA256 hash...`
    );
    fs.copyFileSync(stockElectronExe, targetExe);
  }
};
