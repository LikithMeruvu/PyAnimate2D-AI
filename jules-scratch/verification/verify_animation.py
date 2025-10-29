
import time
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
    page.wait_for_selector("#canvas-container canvas", timeout=60000)

    # Take a series of screenshots
    for i in range(10):
        page.screenshot(path=f"jules-scratch/verification/frame_{i:03d}.png")
        time.sleep(0.1)

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
