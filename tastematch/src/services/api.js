import axios from 'axios';
import * as SecureStore from 'expo-secure-store';
import Config from '../../constants/Config';

// API Configuration is managed in ../constants/Config.ts
const { API_URL } = Config;

const api = axios.create({
    baseURL: API_URL,
    timeout: 10000, // 10 seconds
    headers: {
        'Content-Type': 'application/json',
    },
});

let onUnauthorizedCallback = null;

export const setUnauthorizedCallback = (callback) => {
    console.log('api.js: setUnauthorizedCallback called');
    onUnauthorizedCallback = callback;
};

api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;
        if (error.response.status === 401 && !originalRequest._retry && !originalRequest.url.includes('/auth/refresh')) {
            console.log('api.js: 401 detected, attempting refresh');
            originalRequest._retry = true;
            try {
                const refreshToken = await SecureStore.getItemAsync('refreshToken');
                if (refreshToken) {
                    console.log('api.js: Refresh token found, calling endpoint');
                    const { access_token, refresh_token: newRefreshToken } = await refreshTokenCall(refreshToken);
                    await SecureStore.setItemAsync('userToken', access_token);
                    if (newRefreshToken) {
                        await SecureStore.setItemAsync('refreshToken', newRefreshToken);
                    }
                    api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
                    originalRequest.headers['Authorization'] = `Bearer ${access_token}`;
                    return api(originalRequest);
                } else {
                    console.log('api.js: No refresh token found');
                    throw new Error('No refresh token available');
                }
            } catch (refreshError) {
                console.error('api.js: Token refresh failed:', refreshError);
                await SecureStore.deleteItemAsync('userToken');
                await SecureStore.deleteItemAsync('refreshToken');
                if (onUnauthorizedCallback) {
                    console.log('api.js: Calling onUnauthorizedCallback');
                    onUnauthorizedCallback();
                } else {
                    console.error('api.js: onUnauthorizedCallback is null!');
                }
            }
        }
        return Promise.reject(error);
    }
);

export const login = async (username, password) => {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
};

export const register = async (username, password) => {
    const response = await api.post('/auth/register', { username, password });
    return response.data;
};

export const refreshTokenCall = async (token) => {
    const response = await api.post('/auth/refresh', null, {
        params: { refresh_token: token }
    });
    return response.data;
};

export const getFeed = async (type) => {
    const params = type ? { item_type: type } : {};
    const response = await api.get('/feed/', { params });
    return response.data;
};

export const getMatch = async (matchType) => {
    const response = await api.get('/feed/match', { params: { match_type: matchType } });
    return response.data;
};

export const swipeItem = async (itemId, action) => {
    const response = await api.post('/swipe/', { item_id: itemId, action });
    return response.data;
};

export const getProfile = async () => {
    const response = await api.get('/auth/profile');
    return response.data;
};

export default api;
