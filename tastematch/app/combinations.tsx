import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, FlatList, Image, ActivityIndicator, TouchableOpacity } from 'react-native';
import axios from 'axios';
import Config from '../constants/Config';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

interface ComboItem {
    id: string;
    name: string;
    image_url: string;
    price: number;
    category: string;
    order: number; // 1:Head, 2:Torso, 3:Legs, 4:Feet, 5:Acc
}

interface Combination {
    id: string;
    items: ComboItem[];
    match_score: number;
}

export default function CombinationsScreen() {
    const [combinations, setCombinations] = useState<Combination[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        fetchCombinations();
    }, []);

    const fetchCombinations = async () => {
        try {
            const response = await axios.get(`${Config.API_URL}/outfit/combinations?user_id=test_user`);
            setCombinations(response.data.combinations || []);
        } catch (error) {
            console.error("Error fetching combinations:", error);
        } finally {
            setLoading(false);
        }
    };

    const handlePressCombo = (combo: Combination) => {
        router.push({
            pathname: '/combination_detail',
            params: {
                items: JSON.stringify(combo.items),
                score: combo.match_score
            }
        });
    };

    const renderMannequin = (items: ComboItem[]) => {
        // Sort by visual order
        const sortedItems = [...items].sort((a, b) => a.order - b.order);

        // Group by zone
        const head = sortedItems.filter(i => i.order === 1);
        const torso = sortedItems.filter(i => i.order === 2);
        const legs = sortedItems.filter(i => i.order === 3);
        const feet = sortedItems.filter(i => i.order === 4);
        const accessories = sortedItems.filter(i => i.order === 5);

        return (
            <View style={styles.mannequinContainer}>
                {/* Head Zone */}
                <View style={styles.zoneHead}>
                    {head.map(item => (
                        <Image key={item.id} source={{ uri: item.image_url }} style={styles.imgSmall} />
                    ))}
                </View>

                {/* Torso Zone */}
                <View style={styles.zoneTorso}>
                    {torso.map(item => (
                        <Image key={item.id} source={{ uri: item.image_url }} style={styles.imgLarge} />
                    ))}
                </View>

                {/* Legs Zone */}
                <View style={styles.zoneLegs}>
                    {legs.map(item => (
                        <Image key={item.id} source={{ uri: item.image_url }} style={styles.imgLarge} />
                    ))}
                </View>

                {/* Feet Zone */}
                <View style={styles.zoneFeet}>
                    {feet.map(item => (
                        <Image key={item.id} source={{ uri: item.image_url }} style={styles.imgMedium} />
                    ))}
                </View>

                {/* Accessories Floating */}
                {accessories.length > 0 && (
                    <View style={styles.zoneAccessories}>
                        {accessories.map(item => (
                            <Image key={item.id} source={{ uri: item.image_url }} style={styles.imgSmall} />
                        ))}
                    </View>
                )}
            </View>
        );
    };

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="black" />
                </TouchableOpacity>
                <Text style={styles.title}>Smart Combinations</Text>
            </View>

            {loading ? (
                <View style={styles.center}>
                    <ActivityIndicator size="large" color="#000" />
                    <Text style={{ marginTop: 10 }}>Generating outfits from your likes...</Text>
                </View>
            ) : combinations.length === 0 ? (
                <View style={styles.center}>
                    <Text style={styles.emptyText}>Not enough likes to create combinations.</Text>
                    <Text style={styles.subText}>Swipe more items to get smart suggestions!</Text>
                </View>
            ) : (
                <FlatList
                    data={combinations}
                    keyExtractor={(item) => item.id}
                    renderItem={({ item }) => (
                        <TouchableOpacity onPress={() => handlePressCombo(item)} activeOpacity={0.9}>
                            <View style={styles.comboCard}>
                                <View style={styles.scoreBadge}>
                                    <Text style={styles.scoreText}>{item.match_score}% Match</Text>
                                </View>
                                {renderMannequin(item.items)}
                                <View style={styles.tapHint}>
                                    <Text style={styles.tapText}>Tap to view details</Text>
                                </View>
                            </View>
                        </TouchableOpacity>
                    )}
                    contentContainerStyle={{ paddingBottom: 20 }}
                />
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f8f8f8',
        paddingTop: 50,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 20,
        marginBottom: 20,
    },
    backButton: {
        marginRight: 15,
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    center: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    emptyText: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#555',
    },
    subText: {
        fontSize: 14,
        color: '#888',
        marginTop: 5,
    },
    comboCard: {
        backgroundColor: 'white',
        marginHorizontal: 20,
        marginBottom: 25,
        borderRadius: 20,
        padding: 15,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 8,
        elevation: 5,
        position: 'relative',
        minHeight: 300,
    },
    scoreBadge: {
        position: 'absolute',
        top: 15,
        right: 15,
        backgroundColor: '#e8f5e9',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 15,
        zIndex: 10,
    },
    scoreText: {
        color: 'green',
        fontWeight: 'bold',
        fontSize: 12,
    },
    mannequinContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 10,
    },
    zoneHead: {
        minHeight: 20,
        marginBottom: 5,
        alignItems: 'center',
    },
    zoneTorso: {
        marginBottom: -10, // Overlap slightly
        zIndex: 2,
        alignItems: 'center',
    },
    zoneLegs: {
        marginBottom: -5,
        zIndex: 1,
        alignItems: 'center',
    },
    zoneFeet: {
        marginTop: 5,
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 10,
    },
    zoneAccessories: {
        position: 'absolute',
        right: 10,
        top: 50,
        flexDirection: 'column',
        gap: 5,
    },
    imgSmall: {
        width: 60,
        height: 60,
        resizeMode: 'contain',
    },
    imgMedium: {
        width: 80,
        height: 80,
        resizeMode: 'contain',
    },
    imgLarge: {
        width: 140,
        height: 140,
        resizeMode: 'contain',
    },
    tapHint: {
        alignItems: 'center',
        marginTop: 10,
    },
    tapText: {
        fontSize: 12,
        color: '#aaa',
    }
});
