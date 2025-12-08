import React, { useContext } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { AuthContext } from '../context/AuthContext';

const HomeScreen = () => {
    const { logout } = useContext(AuthContext);

    return (
        <View style={styles.container}>
            <Text style={styles.text}>Welcome to Taste Match!</Text>
            <Text style={styles.subtext}>Auth is working!</Text>
            <Button title="Logout" onPress={logout} />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    text: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    subtext: {
        fontSize: 16,
        color: 'gray',
        marginBottom: 20,
    },
});

export default HomeScreen;
