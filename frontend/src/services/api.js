import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const apiService = {
  predict: async (data, star_info) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/predict/`, {
        data,
        star_info
      });
      return response.data;
    } catch (error) {
      console.error("API Prediction Error:", error);
      throw error;
    }
  }
};

export default apiService;
