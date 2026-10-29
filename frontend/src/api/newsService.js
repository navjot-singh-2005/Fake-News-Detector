import axios from 'axios'

const API_URL = '/api'

export const getRelatedNews = async (query) => {
  if (!query?.trim()) return { articles: [] }
  try {
    const response = await axios.get(`${API_URL}/news`, { params: { q: query.trim() } })
    return response.data
  } catch (error) {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.error ||
      'Related-news search is currently unavailable.'
    throw new Error(message)
  }
}
