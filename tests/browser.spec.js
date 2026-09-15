const { test, expect } = require("@playwright/test");

test("de-identifies locally, withholds review text, and clears state", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator("#engine")).toHaveText("● Local engine ready", {
    timeout: 90000,
  });
  const requests = [];
  page.on("request", (request) =>
    requests.push({
      method: request.method(),
      url: request.url(),
      body: request.postData(),
    }),
  );
  await page.getByRole("button", { name: "De-identify text" }).click();
  await expect(page.locator("#verdict")).toHaveText("Checks passed");
  await expect(page.locator("#output")).toContainText("<CHILD_01>");
  await expect(page.locator("#output")).not.toContainText("Aisha");
  await expect(page.locator("#output")).not.toContainText("parent@example.com");
  expect(requests).toEqual([]);
  await page.screenshot({ path: ".build/desktop.png", fullPage: true });
  expect(
    await page.evaluate(() => [localStorage.length, sessionStorage.length]),
  ).toEqual([0, 0]);

  await page.locator("#example").selectOption("context");
  await page.getByRole("button", { name: "De-identify text" }).click();
  await expect(page.locator("#verdict")).toHaveText("Review required");
  await expect(page.locator("#output")).toContainText(
    "Text withheld for review",
  );
  await page.getByLabel("Show text marked for review").check();
  await expect(page.locator("#output")).toContainText("<AGE_9_11>");

  await page.locator("#input").fill("changed input");
  await expect(page.locator("#output")).toHaveText(
    "Your result will appear here.",
  );
  await expect(page.locator("#copy")).toBeDisabled();
  await page.locator(".known summary").click();
  await page.locator("#known").fill("invalid JSON");
  await page.getByRole("button", { name: "De-identify text" }).click();
  await expect(page.locator("#feedback")).toContainText(
    "must be a JSON object",
  );
  await page.getByRole("button", { name: "Clear all" }).click();
  await expect(page.locator("#input")).toHaveValue("");
  await expect(page.locator("#known")).toHaveValue("{}");
  expect(errors).toEqual([]);
});

test("mobile layout fits the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator("#engine")).toHaveText("● Local engine ready", {
    timeout: 90000,
  });
  await page.getByRole("button", { name: "De-identify text" }).click();
  await expect(page.locator("#verdict")).toHaveText("Checks passed");
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.screenshot({ path: ".build/mobile.png", fullPage: true });
});
