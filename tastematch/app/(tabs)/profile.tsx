import { useFocusEffect } from 'expo-router';
import React, { useContext, useState, useCallback } from 'react';
import { View, Text, StyleSheet, Button, FlatList, Image, ScrollView, RefreshControl } from 'react-native';
import { AuthContext } from '../../src/context/AuthContext';
import { getProfile } from '../../src/services/api';

export default function ProfileScreen() {
    const { logout } = useContext(AuthContext);
    const [profile, setProfile] = useState(null);
    const [refreshing, setRefreshing] = useState(false);

    useFocusEffect(
        useCallback(() => {
            loadProfile();
        }, [])
    );

    const loadProfile = async () => {
        try {
            const data = await getProfile();
            setProfile(data);
        } catch (error) {
            console.error(error);
        }
    };

    const onRefresh = async () => {
        setRefreshing(true);
        await loadProfile();
        setRefreshing(false);
    };

    const renderItem = ({ item }) => (
        <View style={styles.item}>
            <Image source={{ uri: item.image_url }} style={styles.image} />
            <Text numberOfLines={1} style={styles.itemTitle}>{item.title}</Text>
        </View>
    );

    if (!profile) return <View style={styles.container}><Text>Loading...</Text></View>;

    return (
        <ScrollView
            style={styles.container}
            refreshControl={
                <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
        >
            <View style={styles.header}>
                <Text style={styles.username}>@{profile.username}</Text>
                <Button title="Logout" onPress={logout} color="red" />
            </View>

            <View style={styles.section}>
                <Text style={styles.sectionTitle}>🔥 Super Likes</Text>
                {profile.superlikes.length === 0 ? (
                    <Text style={styles.emptyText}>No super likes yet. Swipe Up!</Text>
                ) : (
                    <FlatList
                        horizontal
                        data={profile.superlikes}
                        renderItem={renderItem}
                        keyExtractor={(item) => item.id.toString()}
                        showsHorizontalScrollIndicator={false}
                    />
                )}
            </View>

            <View style={styles.section}>
                <Text style={styles.sectionTitle}>👀 Watchlist</Text>
                {profile.watchlist.length === 0 ? (
                    <Text style={styles.emptyText}>Watchlist is empty. Swipe Down!</Text>
                ) : (
                    <FlatList
                        horizontal
                        data={profile.watchlist}
                        renderItem={renderItem}
                        keyExtractor={(item) => item.id.toString()}
                        showsHorizontalScrollIndicator={false}
                    />
                )}
            </View>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        padding: 20,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 30,
        marginTop: 40,
    },
    username: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    section: {
        marginBottom: 30,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 10,
    },
    emptyText: {
        color: '#888',
        fontStyle: 'italic',
    },
    item: {
        marginRight: 15,
        width: 100,
    },
    image: {
        width: 100,
        height: 150,
        borderRadius: 10,
        marginBottom: 5,
    },
    itemTitle: {
        fontSize: 12,
        textAlign: 'center',
    },
});
