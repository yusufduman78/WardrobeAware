import React, { createContext, useState, useEffect } from 'react';
import * as SecureStore from 'expo-secure-store';
import { useRouter } from 'expo-router';
import { login as apiLogin, register as apiRegister, setUnauthorizedCallback } from '../services/api';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [userToken, setUserToken] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        const bootstrapAsync = async () => {
            let token;
            try {
                token = await SecureStore.getItemAsync('userToken');
            } catch (e) {
                console.log('Restoring token failed', e);
            }
            setUserToken(token);
            setIsLoading(false);
        };

        bootstrapAsync();

        // Register logout callback to handle 401s from api.js
        console.log('AuthContext: Registering unauthorized callback');
        setUnauthorizedCallback(() => {
            console.log("AuthContext: Unauthorized callback triggered! Logging out...");
            logout();
        });
    }, []);

    const login = async (username, password) => {
        try {
            const data = await apiLogin(username, password);
            await SecureStore.setItemAsync('userToken', data.access_token);
            if (data.refresh_token) {
                await SecureStore.setItemAsync('refreshToken', data.refresh_token);
            }
            setUserToken(data.access_token);
        } catch (e) {
            console.error(e);
            throw e;
        }
    };

    const register = async (username, password) => {
        try {
            const data = await apiRegister(username, password);
            // Auto-login after register if backend returns tokens
            if (data.access_token) {
                await SecureStore.setItemAsync('userToken', data.access_token);
                if (data.refresh_token) {
                    await SecureStore.setItemAsync('refreshToken', data.refresh_token);
                }
                setUserToken(data.access_token);
            }
        } catch (e) {
            console.error(e);
            throw e;
        }
    };

    const logout = async () => {
        try {
            await SecureStore.deleteItemAsync('userToken');
            await SecureStore.deleteItemAsync('refreshToken');
        } catch (e) {
            console.error("Error clearing tokens", e);
        }
        setUserToken(null);
        // Force navigation to login
        if (router) {
            router.replace('/login');
        }
    };

    const authContext = {
        userToken,
        isLoading,
        login,
        register,
        logout,
    };

    return (
        <AuthContext.Provider value={authContext}>
            {children}
        </AuthContext.Provider>
    );
};
