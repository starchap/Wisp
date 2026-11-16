const { execFileSync } = require('child_process');
const path = require('path');

exports.default = async function afterSign(context) {
  if (context.electronPlatformName !== 'darwin') {
    return;
  }
  const appName = context.packager.appInfo.productFilename;
  const appPath = path.join(context.appOutDir, `${appName}.app`);
  try {
    console.log(`[afterSign] Applying deep ad-hoc signature to ${appPath} ...`);
    execFileSync('codesign', ['--force', '--deep', '--sign', '-', appPath], {
      stdio: 'inherit',
    });
  } catch (err) {
    console.warn('[afterSign] codesign warning:', err.message);
  }
};
