import axios from 'axios'

const API_URL = '/api'

export const analyzeUrl = async (url, text) => {
  try {
    const payload = {
      url: url ? url.trim() || null : null,
      text: text ? text.trim() || null : null,
    }
    const response = await axios.post(`${API_URL}/analyze`, payload)
    return response.data
  } catch (error) {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.error ||
      'Failed to analyze the article. Check that the backend and trained ML service are available.'
    throw new Error(message)
  }
}
