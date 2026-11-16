const express = require('express');
const axios = require('axios');
const cors = require('cors');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { Mutex } = require('async-mutex');
const { HttpsProxyAgent } = require('https-proxy-agent');
const { SocksProxyAgent } = require('socks-proxy-agent');
const fs = require('fs');
const path = require('path');
const os = require('os');

puppeteer.use(StealthPlugin());

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Browser instance
let browser = null;
let browserMutex = new Mutex();
const MAX_BROWSER_AGE = 180000; // 3 minutes

// User agents
const USER_AGENTS = [
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
];

// Proxy list
const PROXIES = [
  // 'http://user:pass@host:port'
];

// Helpers
function getRandomUserAgent() {
  return USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
}
function getRandomProxy() {
  return PROXIES.length ? PROXIES[Math.floor(Math.random() * PROXIES.length)] : null;
}

// Initialize Puppeteer
async function initBrowser() {
  if (browser) {
    try { await browser.close(); } catch (e) { console.error(e.message); }
  }

  const launchOptions = {
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-web-security',
      '--disable-features=IsolateOrigins,site-per-process',
      '--disable-blink-features=AutomationControlled',
      '--window-size=1920,1080'
    ],
    ignoreHTTPSErrors: true
  };

  const proxy = getRandomProxy();
  if (proxy) launchOptions.args.push(`--proxy-server=${proxy}`);

  browser = await puppeteer.launch(launchOptions);
  browser.createdAt = Date.now();
  console.log('Browser initialized');
}

// Get Puppeteer page
async function getBrowserPage() {
  const release = await browserMutex.acquire();
  try {
    if (!browser || Date.now() - browser.createdAt > MAX_BROWSER_AGE) await initBrowser();
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.setUserAgent(getRandomUserAgent());
    await page.setRequestInterception(true);

    page.on('request', req => {
      if (['image', 'media', 'font'].includes(req.resourceType())) req.continue();
      else req.abort();
    });

    return page;
  } finally { release(); }
}

// Fetch image with Puppeteer
async function fetchWithBrowser(imageUrl) {
  let page = null;
  try {
    page = await getBrowserPage();
    const response = await page.goto(imageUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
    if (!response || response.status() !== 200) throw new Error(`Status: ${response ? response.status() : 'no response'}`);

    const base64 = await page.evaluate(() => {
      const img = document.querySelector('img');
      if (!img) return null;
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      return canvas.toDataURL('image/png').split(',')[1];
    });

    if (!base64) throw new Error('Could not extract image');
    return base64;
  } finally {
    if (page) try { await page.close(); } catch (e) { console.error(e.message); }
  }
}

// Fetch image via proxy
async function fetchWithProxy(imageUrl) {
  const proxy = getRandomProxy();
  if (!proxy) throw new Error('No proxy available');

  const agent = proxy.startsWith('socks') ? new SocksProxyAgent(proxy) : new HttpsProxyAgent(proxy);
  const response = await axios.get(imageUrl, {
    responseType: 'arraybuffer',
    timeout: 15000,
    httpAgent: agent,
    httpsAgent: agent,
    headers: { 'User-Agent': getRandomUserAgent(), 'Referer': new URL(imageUrl).origin }
  });

  if (!response.headers['content-type']?.startsWith('image/')) throw new Error('Not an image');
  return Buffer.from(response.data, 'binary').toString('base64');
}

// Download image directly
async function downloadImageToBase64(imageUrl) {
  const response = await axios.get(imageUrl, {
    responseType: 'arraybuffer',
    timeout: 15000,
    headers: { 'User-Agent': getRandomUserAgent(), 'Referer': new URL(imageUrl).origin }
  });
  if (!response.headers['content-type']?.startsWith('image/')) throw new Error('Not an image');

  // Optional: save to temp file
  const tmpPath = path.join(os.tmpdir(), `${Date.now()}.jpg`);
  fs.writeFileSync(tmpPath, response.data);

  return Buffer.from(response.data, 'binary').toString('base64');
}

// Endpoint
app.post('/api/image-to-base64', async (req, res) => {
  const { imageUrl } = req.body;
  if (!imageUrl) return res.status(400).json({ error: 'Missing imageUrl' });

  console.log('Fetching image:', imageUrl);

  // Try strategies in order
  try { return res.json({ base64: await downloadImageToBase64(imageUrl) }); } catch (e) { console.error('Download failed:', e.message); }
  try { return res.json({ base64: await fetchWithProxy(imageUrl) }); } catch (e) { console.error('Proxy failed:', e.message); }
  try { return res.json({ base64: await fetchWithBrowser(imageUrl) }); } catch (e) { console.error('Browser failed:', e.message); }

  res.status(500).json({ error: 'All methods failed' });
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString(), browser: browser ? 'Connected' : 'Disconnected' });
});

// Start server
app.listen(PORT, '0.0.0.0', () => console.log(`CORS Proxy running at http://0.0.0.0:${PORT}`));

// Cleanup
process.on('SIGINT', async () => { if (browser) await browser.close(); process.exit(0); });
