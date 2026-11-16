const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

exports.default = async function afterSign(context) {
  if (context.electronPlatformName !== 'darwin') {
    return;
  }
  const appName = context.packager.appInfo.productFilename;
  const appPath = path.join(context.appOutDir, `${appName}.app`);
  const backendBin = path.join(
    appPath,
    'Contents',
    'Resources',
    'backend-dist',
    'wisp-backend',
    'wisp-backend'
  );

  if (fs.existsSync(backendBin)) {
    console.log(`[afterSign] Signing embedded backend binary: ${backendBin}`);
    execFileSync('chmod', ['+x', backendBin], { stdio: 'inherit' });
    execFileSync('codesign', ['--force', '--sign', '-', backendBin], {
      stdio: 'inherit',
    });
  }

  console.log(`[afterSign] Applying deep ad-hoc signature to ${appPath} ...`);
  execFileSync('codesign', ['--force', '--deep', '--sign', '-', appPath], {
    stdio: 'inherit',
  });

  console.log(`[afterSign] Verifying strict macOS bundle signature for ${appPath} ...`);
  execFileSync(
    'codesign',
    ['--verify', '--deep', '--strict', '--verbose=2', appPath],
    { stdio: 'inherit' }
  );
};
