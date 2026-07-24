import { test, expect, type Page } from "@playwright/test";

/**
 * Smoke tests against the real app (baseline model, no weights):
 * upload the demo CSV → tidy the messy header → mark the target → predict /
 * compare, with the plain-language metric explanations visible.
 */

async function setUpDemoPrediction(page: Page) {
  await page.goto("/");
  await page.getByText("Simple — sales report (CSV)").click();

  // Wait for the grid; row 4 holds the real header (region, …, revenue).
  await page.getByRole("rowheader", { name: /^4( header)?$/ }).waitFor();

  // Fresh project: apply the header. A reopened project already has it.
  const revenueHeader = page.getByRole("columnheader", { name: /^revenue/ });
  if ((await revenueHeader.count()) === 0) {
    await page.getByRole("rowheader", { name: "4", exact: true }).click();
    await page.getByRole("menu").getByText("Use as header row").click();
    await revenueHeader.waitFor();
  }

  // Mark the target and use the fast baseline model.
  await revenueHeader.click();
  await page.getByRole("menuitemradio", { name: /Predict this column/ }).click();
  await page.locator("select").first().selectOption("baseline");
}

test("demo upload → tidy → predict → explained metrics", async ({ page }) => {
  await setUpDemoPrediction(page);
  await page.getByRole("button", { name: /^Predict( again)?$/ }).click();

  // Results: metric tiles carry plain-language tooltips.
  const tile = page.locator(".tile[data-tip]").first();
  await expect(tile).toBeVisible({ timeout: 90_000 });
  await tile.hover();
  await expect
    .poll(async () => tile.evaluate((el) => getComputedStyle(el, "::after").visibility))
    .toBe("visible");

  // The demo's 25 labeled rows leave only 5 held out → the rough-check warning.
  await expect(page.getByText(/held out for the accuracy check/).first()).toBeVisible();

  // The thorough check runs 5-fold cross-validation and reports mean ± spread.
  await page.getByRole("button", { name: /thorough check/i }).click();
  await expect(page.getByText(/across 5 folds/)).toBeVisible({ timeout: 90_000 });

  // Per-row explanation: ask why for the first predicted row.
  await page.getByRole("button", { name: /why this prediction/i }).click();
  await expect(page.getByText(/with a typical/i).first()).toBeVisible({ timeout: 90_000 });
});

test("fill every empty cell (impute)", async ({ page }) => {
  await setUpDemoPrediction(page);
  await page.getByRole("button", { name: /fill every empty cell/i }).click();
  await expect(page.getByText(/Filled \d+ cell/)).toBeVisible({ timeout: 90_000 });
});

test("compare models includes the classic-ML baseline", async ({ page }) => {
  await setUpDemoPrediction(page);
  await page.getByRole("button", { name: "Compare models" }).click();

  // The baseline is a first-class reference column; foundation models
  // without weights show error notes instead of blocking the table.
  await expect(
    page.locator(".compare-table th", { hasText: /classic ML/ }),
  ).toBeVisible({ timeout: 120_000 });
});
