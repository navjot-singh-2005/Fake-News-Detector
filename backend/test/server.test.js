const { after, before, test } = require('node:test');
const assert = require('node:assert/strict');
process.env.ML_SERVICE_URL = 'http://127.0.0.1:1';
const { app } = require('../server');

let server;
let baseUrl;

before(async () => {
  server = app.listen(0);
  await new Promise(resolve => server.once('listening', resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
});

test('rejects missing prediction text', async () => {
  const response = await fetch(`${baseUrl}/api/predict`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: '  ' }),
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /text is required/i);
});

test('rejects excessively long prediction text', async () => {
  const response = await fetch(`${baseUrl}/api/predict`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: 'a'.repeat(100_001) }),
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /100000 characters/);
});

test('rejects malformed JSON with a JSON error response', async () => {
  const response = await fetch(`${baseUrl}/api/predict`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{"text":',
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /valid JSON/);
});

test('rejects non-HTTP URL protocols and does not classify without text', async () => {
  const response = await fetch(`${baseUrl}/api/analyze`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ url: 'file:///article.txt', text: 'An article headline with enough text.' }),
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /HTTP or HTTPS/);
});

test('returns explicit service-unavailable response when the Python model API is down', async () => {
  const response = await fetch(`${baseUrl}/api/predict`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: 'An article headline with enough text.' }),
  });
  assert.equal(response.status, 503);
  assert.match((await response.json()).error, /Python ML service is unavailable/);
});
