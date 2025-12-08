import React from 'react';
import { View, Text, StyleSheet, Image, ScrollView, TouchableOpacity } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

export default function CombinationDetailScreen() {
    const params = useLocalSearchParams();
    const router = useRouter();

    // Parse the items string back to object
    const items = params.items ? JSON.parse(params.items as string) : [];
    const score = params.score;

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="black" />
                </TouchableOpacity>
                <Text style={styles.title}>Outfit Details</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.scoreContainer}>
                    <Text style={styles.scoreText}>{score}% Match</Text>
                    <Text style={styles.scoreSubtext}>Based on visual compatibility</Text>
                </View>

                <View style={styles.itemsList}>
                    {items.map((item: any, index: number) => (
                        <View key={index} style={styles.itemCard}>
                            <Image source={{ uri: item.image_url }} style={styles.itemImage} />
                            <View style={styles.itemInfo}>
                                <Text style={styles.itemName}>{item.name}</Text>
                                <Text style={styles.itemCategory}>{item.category?.toUpperCase()}</Text>
                                <Text style={styles.itemPrice}>${item.price}</Text>
                                <TouchableOpacity style={styles.buyButton}>
                                    <Text style={styles.buyButtonText}>Shop Now</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    ))}
                </View>
            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        paddingTop: 50,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 20,
        marginBottom: 20,
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
        paddingBottom: 10,
    },
    backButton: {
        marginRight: 15,
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
    },
    scrollContent: {
        paddingBottom: 40,
    },
    scoreContainer: {
        alignItems: 'center',
        marginBottom: 20,
        backgroundColor: '#f0fdf4',
        padding: 15,
        marginHorizontal: 20,
        borderRadius: 10,
        borderWidth: 1,
        borderColor: '#dcfce7',
    },
    scoreText: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#166534',
    },
    scoreSubtext: {
        fontSize: 12,
        color: '#15803d',
        marginTop: 5,
    },
    itemsList: {
        paddingHorizontal: 20,
    },
    itemCard: {
        flexDirection: 'row',
        marginBottom: 20,
        backgroundColor: '#fff',
        borderRadius: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 5,
        elevation: 2,
        padding: 10,
        borderWidth: 1,
        borderColor: '#f0f0f0',
    },
    itemImage: {
        width: 100,
        height: 100,
        borderRadius: 8,
        backgroundColor: '#f9f9f9',
    },
    itemInfo: {
        flex: 1,
        marginLeft: 15,
        justifyContent: 'center',
    },
    itemName: {
        fontSize: 16,
        fontWeight: '600',
        marginBottom: 5,
        color: '#333',
    },
    itemCategory: {
        fontSize: 12,
        color: '#888',
        marginBottom: 5,
    },
    itemPrice: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#000',
        marginBottom: 10,
    },
    buyButton: {
        backgroundColor: '#000',
        paddingVertical: 8,
        paddingHorizontal: 15,
        borderRadius: 20,
        alignSelf: 'flex-start',
    },
    buyButtonText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: 'bold',
    },
});
