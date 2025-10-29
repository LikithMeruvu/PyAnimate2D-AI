
from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("http://127.0.0.1:5000")

    # Fill in the API key and prompt
    page.fill("#api_key", "test-key")
    page.fill("#prompt", "A simple animation of a bouncing ball.")

    # Click the generate button and wait for the animation to load
    page.click("button[type='submit']")
    page.wait_for_selector("#canvas-container canvas")

    # Take a screenshot
    page.screenshot(path="jules-scratch/verification/verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
