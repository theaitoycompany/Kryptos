const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests",
  testMatch: "browser.spec.js",
  timeout: 120000,
  workers: 1,
  use: {
    baseURL: process.env.KRYPTOS_DEMO_URL || "http://127.0.0.1:8765",
    browserName: "chromium",
  },
  webServer: process.env.KRYPTOS_DEMO_URL
    ? undefined
    : {
        command:
          "python3 -m http.server 8765 --bind 127.0.0.1 --directory .build/space",
        url: "http://127.0.0.1:8765",
        reuseExistingServer: !process.env.CI,
      },
});
