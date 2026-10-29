require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 5001;
const ML_SERVICE_URL = (process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001').replace(/\/$/, '');
const MAX_TEXT_LENGTH = 100_000;

app.use(cors());
app.use(express.json({ limit: '1mb' }));

function validateText(text) {
  if (typeof text !== 'string' || !text.trim()) {
    return 'Article text is required.';
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return `Article text must be no longer than ${MAX_TEXT_LENGTH} characters.`;
  }
  return null;
}

function extractUrlFeatures(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return null;
  }
  if (!['http:', 'https:'].includes(parsed.protocol)) return null;

  const domain = parsed.hostname.toLowerCase();
  const suspiciousTlds = ['.xyz', '.click', '.info', '.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.buzz', '.work'];
  const suspiciousKeywords = ['clickbait', 'shocking', 'viral', 'exposed', 'truth', 'secret', 'miracle', 'alert', 'breaking', 'urgent', 'weird'];
  const recognizedNewsDomains = ['bbc.com', 'reuters.com', 'apnews.com', 'theguardian.com', 'nytimes.com', 'washingtonpost.com', 'bloomberg.com', 'npr.org', 'thehindu.com', 'ndtv.com', 'hindustantimes.com', 'timesofindia.com', 'aljazeera.com'];
  const isHttps = parsed.protocol === 'https:';
  const isRecognizedNewsDomain = recognizedNewsDomains.some(
    known => domain === known || domain.endsWith(`.${known}`)
  );
  const hasSuspiciousTLD = suspiciousTlds.some(tld => domain.endsWith(tld));
  const foundKeywords = suspiciousKeywords.filter(keyword => rawUrl.toLowerCase().includes(keyword));
  const pathDepth = parsed.pathname.split('/').filter(Boolean).length;
  const flags = [];

  if (!isHttps) flags.push('The URL does not use HTTPS encryption.');
  if (hasSuspiciousTLD) flags.push('The URL uses a commonly abused top-level domain.');
  if (foundKeywords.length) flags.push(`Potentially sensational URL terms: ${foundKeywords.join(', ')}.`);
  if (domain.length > 30) flags.push('The domain name is unusually long.');
  let trustScore = 50 + (isHttps ? 15 : -15) + (isRecognizedNewsDomain ? 20 : 0);
  if (hasSuspiciousTLD) trustScore -= 15;
  trustScore -= foundKeywords.length * 5;
  if (domain.length > 30) trustScore -= 5;

  return {
    domain,
    isHttps,
    isTrusted: isRecognizedNewsDomain,
    domainAge: null,
    isBlacklisted: null,
    urlLength: rawUrl.length,
    pathDepth,
    hasSuspiciousTLD,
    suspiciousKeywords: foundKeywords,
    trustScore: Math.max(0, Math.min(100, Math.round(trustScore))),
    flags,
  };
}

function makeExplanation(label, modelName, hasUrl, signals) {
  const verdict = label === 'REAL' ? 'classified as REAL' : 'classified as FAKE';
  let exp = `The ${modelName} model ${verdict} based on the supplied text.`;
  const keyTerms = signals ? (label === 'REAL' ? signals.real : signals.fake) : [];
  if (keyTerms && keyTerms.length) {
    exp += ` Key indicative terms identified: ${keyTerms.join(', ')}.`;
  }
  if (hasUrl) {
    exp += ' URL indicators are separate heuristics and do not establish whether the article is true.';
  }
  exp += ' This machine-learning result is not definitive proof; verify important claims with reliable independent sources.';
  return exp;
}

function getModelConfidence(prediction) {
  if (!prediction.probabilities || typeof prediction.probabilities !== 'object') return null;
  const probabilities = Object.values(prediction.probabilities).filter(Number.isFinite);
  if (!probabilities.length) return null;
  return Math.round(Math.max(...probabilities) * 100);
}

async function requestPrediction(text) {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/api/predict`, { text }, { timeout: 30_000 });
    return response.data;
  } catch (error) {
    const status = error.response?.status || 503;
    const detail = error.response?.data?.detail || 'The Python ML service is unavailable.';
    const apiError = new Error(detail);
    apiError.status = status;
    throw apiError;
  }
}

app.post('/api/predict', async (req, res) => {
  const validationError = validateText(req.body?.text);
  if (validationError) return res.status(400).json({ error: validationError });
  try {
    return res.json(await requestPrediction(req.body.text));
  } catch (error) {
    return res.status(error.status || 503).json({ error: error.message });
  }
});

async function requestAnalysis(url, text) {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/api/analyze`, { url: url || null, text: text || null }, { timeout: 35_000 });
    return response.data;
  } catch (error) {
    const status = error.response?.status || 503;
    const detail = error.response?.data?.detail || error.response?.data?.error || 'The Python ML service is unavailable.';
    const apiError = new Error(detail);
    apiError.status = status;
    throw apiError;
  }
}

app.post('/api/analyze', async (req, res) => {
  const { url: rawUrl, text } = req.body || {};
  const hasText = Boolean(typeof text === 'string' && text.trim());
  const hasUrl = Boolean(typeof rawUrl === 'string' && rawUrl.trim());

  if (!hasText && !hasUrl) {
    return res.status(400).json({ error: 'Article text or a valid article URL is required.' });
  }

  if (hasText && text.length > MAX_TEXT_LENGTH) {
    return res.status(400).json({ error: `Article text must be no longer than ${MAX_TEXT_LENGTH} characters.` });
  }

  let urlFeatures = null;
  if (hasUrl) {
    urlFeatures = extractUrlFeatures(rawUrl.trim());
    if (!urlFeatures) {
      return res.status(400).json({ error: 'Invalid URL. Enter a valid HTTP or HTTPS URL.' });
    }
  }

  try {
    const analysis = await requestAnalysis(rawUrl, text);
    return res.json(analysis);
  } catch (error) {
    return res.status(error.status || 503).json({ error: error.message });
  }
});

app.get('/api/models', async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/api/models`, { timeout: 10_000 });
    return res.json(response.data);
  } catch (error) {
    return res.status(error.response?.status || 503).json({ error: error.message });
  }
});

app.get('/api/news', async (req, res) => {
  const query = req.query.q || req.query.query;
  if (typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ error: 'A search query is required.' });
  }
  const apiKey = process.env.NEWS_API_KEY;
  if (!apiKey) {
    return res.status(503).json({ error: 'Related-news search is unavailable; configure NEWS_API_KEY to enable it.' });
  }
  try {
    const response = await axios.get('https://newsapi.org/v2/everything', {
      params: { q: query.trim(), language: 'en', pageSize: 5, sortBy: 'relevancy', apiKey },
      timeout: 6000,
    });
    const articles = (response.data.articles || []).map(article => ({
      title: article.title,
      source: article.source?.name || 'Unknown',
      url: article.url,
    }));
    return res.json({ articles });
  } catch (error) {
    console.error('Related-news request failed:', error.message);
    return res.status(502).json({ error: 'Related-news provider request failed.' });
  }
});

app.get('/api/factcheck', async (req, res) => {
  const query = req.query.q;
  if (typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ error: 'A search query is required.' });
  }
  const apiKey = process.env.FACT_CHECK_API_KEY;
  if (!apiKey) {
    return res.status(503).json({ error: 'Fact-check search is unavailable; configure FACT_CHECK_API_KEY to enable it.' });
  }
  try {
    const response = await axios.get(
      'https://factchecktools.googleapis.com/v1alpha1/claims:search',
      { params: { query: query.trim(), key: apiKey }, timeout: 6000 }
    );
    return res.json({ claims: response.data.claims || [] });
  } catch (error) {
    console.error('Fact-check request failed:', error.message);
    return res.status(502).json({ error: 'Fact-check provider request failed.' });
  }
});

app.get('/api/health', async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 3000 });
    const status = response.data.model_loaded ? 200 : 503;
    return res.status(status).json({
      status: response.data.model_loaded ? 'ok' : 'degraded',
      service: 'TruthLens API',
      ml: response.data,
    });
  } catch {
    return res.status(503).json({
      status: 'degraded',
      service: 'TruthLens API',
      ml: { status: 'unavailable', model_loaded: false },
    });
  }
});

app.use((error, req, res, next) => {
  if (error instanceof SyntaxError && error.status === 400 && 'body' in error) {
    return res.status(400).json({ error: 'Request body must contain valid JSON.' });
  }
  console.error('Unhandled API error:', error);
  return res.status(500).json({ error: 'An unexpected server error occurred.' });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`TruthLens API running on http://localhost:${PORT}`);
    console.log('Endpoints: POST /api/predict | POST /api/analyze | GET /api/news | GET /api/factcheck');
  });
}

module.exports = { app, extractUrlFeatures };
