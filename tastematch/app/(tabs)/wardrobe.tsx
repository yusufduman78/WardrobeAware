import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Image, FlatList, TouchableOpacity, ActivityIndicator, Modal, ScrollView, Alert } from 'react-native';
import axios from 'axios';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import Config from '../../constants/Config';

interface Item {
    id: string;
    name: string;
    price: number;
    image_url: string;
    match_score?: number;
    category?: string;
}

export default function WardrobeScreen() {
    const [selectedItem, setSelectedItem] = useState<Item | null>(null);
    const [recommendations, setRecommendations] = useState<Item[]>([]);
    const [loading, setLoading] = useState(false);
    const [loadingText, setLoadingText] = useState("Loading...");
    const [progress, setProgress] = useState(0);
    const [showConfirmation, setShowConfirmation] = useState(false);
    const [confirmationData, setConfirmationData] = useState<any>(null); // Temp data for upload
    const [tempCategory, setTempCategory] = useState("");
    const [availableCategories, setAvailableCategories] = useState<string[]>([]);

    const [myItems, setMyItems] = useState<Item[]>([]); // Mock "My Wardrobe"
    const router = useRouter();

    useEffect(() => {
        // Load some random items as "My Wardrobe"
        fetchMyItems();
    }, []);

    // Simulate Progress Bar with Dynamic Text
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (loading) {
            setProgress(0);
            interval = setInterval(() => {
                setProgress((prev) => {
                    if (prev >= 98) return prev; // Cap at 98%

                    // Smoother, slower increment
                    // Start fastish, then slow down significantly
                    const increment = prev < 30 ? 2 : prev < 70 ? 1 : prev < 90 ? 0.5 : 0.2;
                    const nextProgress = prev + increment;

                    // Dynamic Text Updates based on progress
                    if (nextProgress > 90) {
                        setLoadingText("Almost there! Polishing results... ✨");
                    } else if (nextProgress > 70) {
                        setLoadingText("Mixing and matching items... 🧥");
                    } else if (nextProgress > 40) {
                        setLoadingText("Finding the best color combinations... 🎨");
                    }
                    // Note: Initial text is set by the calling function (upload/recommend)

                    return nextProgress;
                });
            }, 100); // Faster tick rate for smoother animation
        } else {
            setProgress(100);
        }
        return () => clearInterval(interval);
    }, [loading]);

    const fetchMyItems = async () => {
        try {
            const response = await axios.get(`${Config.API_URL}/feed?limit=5`);
            setMyItems(response.data);
        } catch (error) {
            console.error("Error fetching wardrobe:", error);
        }
    };

    const getRecommendations = async (item: Item) => {
        setSelectedItem(item);
        setLoading(true);
        setLoadingText("Analyzing style & finding matches...");
        try {
            const response = await axios.post(`${Config.API_URL}/outfit/complete`, {
                item_id: item.id
            });
            // Backend returns { recommendations: [...] }
            setRecommendations(response.data.recommendations || []);
        } catch (error) {
            console.error("Error getting recommendations:", error);
        } finally {
            setLoading(false);
        }
    };

    const createOutfit = async () => {
        if (!selectedItem) return;
        setLoading(true);
        setLoadingText("Designing full outfit...");
        try {
            const response = await axios.post(`${Config.API_URL}/outfit/create_from_anchor`, {
                anchor_item_id: selectedItem.id
            });
            const combo = response.data;
            if (combo) {
                router.push({
                    pathname: '/combination_detail',
                    params: {
                        items: JSON.stringify(combo.items),
                        score: combo.match_score
                    }
                });
            }
        } catch (error) {
            console.error("Error creating outfit:", error);
            alert("Could not create outfit for this item.");
        } finally {
            setLoading(false);
        }
    };

    const pickImage = async () => {
        // No permission request needed for launching the library
        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [1, 1],
            quality: 1,
        });

        if (!result.canceled) {
            uploadImage(result.assets[0].uri);
        }
    };

    const uploadImage = async (uri: string) => {
        setLoading(true);
        setLoadingText("Uploading & Removing Background...");
        const formData = new FormData();

        // Append file
        const filename = uri.split('/').pop();
        const match = /\.(\w+)$/.exec(filename || '');
        const type = match ? `image/${match[1]}` : `image`;

        // @ts-ignore
        formData.append('file', { uri, name: filename, type });
        formData.append('user_id', 'test_user');

        try {
            const response = await axios.post(`${Config.API_URL}/wardrobe/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            // Instead of auto-adding, open confirmation modal
            const data = response.data;
            setConfirmationData(data);
            setTempCategory(data.predicted_category || "tops");
            setAvailableCategories(data.all_categories || []);
            setShowConfirmation(true);

            // setMyItems(prev => [response.data, ...prev]);
            // alert("Item uploaded and background removed! ✨");
        } catch (error) {
            console.error("Error uploading image:", error);
            alert("Upload failed.");
        } finally {
            setLoading(false);
        }
    };

    const confirmCategory = async () => {
        if (!confirmationData) return;

        setLoading(true);
        setLoadingText("Saving item...");

        try {
            // Update category on backend
            await axios.post(`${Config.API_URL}/wardrobe/update_category`, {
                item_id: confirmationData.id,
                category: tempCategory
            });

            // Add to local list with confirmed category
            const newItem = { ...confirmationData, category: tempCategory };
            setMyItems(prev => [newItem, ...prev]);

            setShowConfirmation(false);
            setConfirmationData(null);
            alert("Item saved to your wardrobe! 🧥");

        } catch (error) {
            console.error("Error confirming category:", error);
            alert("Failed to save category.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <View style={styles.container}>
            <View style={styles.headerContainer}>
                <Text style={styles.header}>My Wardrobe</Text>
                <View style={{ flexDirection: 'row', gap: 10 }}>
                    <TouchableOpacity
                        style={styles.iconButton}
                        onPress={pickImage}
                    >
                        <Ionicons name="camera" size={24} color="white" />
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.comboButton}
                        onPress={() => router.push('/combinations')}
                    >
                        <Text style={styles.comboButtonText}>Smart Combos ✨</Text>
                    </TouchableOpacity>
                </View>
            </View>

            <View style={styles.wardrobeList}>
                <FlatList
                    data={myItems}
                    horizontal
                    keyExtractor={item => item.id}
                    renderItem={({ item }) => (
                        <TouchableOpacity onPress={() => getRecommendations(item)} style={styles.wardrobeItem}>
                            <Image source={{ uri: item.image_url }} style={styles.wardrobeImage} />
                        </TouchableOpacity>
                    )}
                />
            </View>

            <View style={styles.divider} />

            <View style={styles.sectionHeader}>
                <Text style={styles.header}>Complete the Look</Text>
                {selectedItem && (
                    <TouchableOpacity style={styles.createButton} onPress={createOutfit}>
                        <Text style={styles.createButtonText}>Create Full Outfit 🪄</Text>
                    </TouchableOpacity>
                )}
            </View>

            {/* Replaced inline loader with Modal below */}
            {selectedItem ? (
                <FlatList
                    data={recommendations}
                    numColumns={2}
                    keyExtractor={item => item.id}
                    renderItem={({ item }) => (
                        <View style={styles.recItem}>
                            <Image source={{ uri: item.image_url }} style={styles.recImage} />
                            <Text style={styles.recName} numberOfLines={1}>{item.name}</Text>
                            <Text style={styles.recCategory}>{item.category?.toUpperCase() || 'ITEM'}</Text>
                            <Text style={styles.recScore}>Match: {item.match_score}%</Text>
                        </View>
                    )}
                />
            ) : (
                <Text style={styles.placeholder}>Select an item from your wardrobe to see recommendations.</Text>
            )}
            {/* Loading Overlay with Progress Bar */}
            <Modal transparent={true} visible={loading} animationType="fade">
                <View style={styles.loadingOverlay}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator size="large" color="#7c3aed" />
                        <Text style={styles.loadingText}>{loadingText}</Text>

                        {/* Progress Bar */}
                        <View style={styles.progressBarContainer}>
                            <View style={[styles.progressBarFill, { width: `${progress}%` }]} />
                        </View>
                        <Text style={styles.progressText}>{Math.round(progress)}%</Text>
                    </View>
                </View>
            </Modal>

            {/* Category Confirmation Modal */}
            <Modal transparent={true} visible={showConfirmation} animationType="slide">
                <View style={styles.modalOverlay}>
                    <View style={styles.confirmationContainer}>
                        <Text style={styles.modalTitle}>Confirm Category</Text>

                        {confirmationData && (
                            <Image source={{ uri: confirmationData.image_url }} style={styles.previewImage} />
                        )}

                        <Text style={styles.modalSubtitle}>
                            We think this is a <Text style={{ fontWeight: 'bold', color: '#7c3aed' }}>{tempCategory.toUpperCase()}</Text>. Is that correct?
                        </Text>

                        <View style={{ height: 150, marginVertical: 10 }}>
                            <ScrollView nestedScrollEnabled>
                                <View style={styles.categoryList}>
                                    {availableCategories.map((cat) => (
                                        <TouchableOpacity
                                            key={cat}
                                            style={[
                                                styles.categoryChip,
                                                tempCategory === cat && styles.selectedCategoryChip
                                            ]}
                                            onPress={() => setTempCategory(cat)}
                                        >
                                            <Text style={[
                                                styles.categoryChipText,
                                                tempCategory === cat && styles.selectedCategoryChipText
                                            ]}>{cat}</Text>
                                        </TouchableOpacity>
                                    ))}
                                </View>
                            </ScrollView>
                        </View>

                        <TouchableOpacity style={styles.confirmButton} onPress={confirmCategory}>
                            <Text style={styles.confirmButtonText}>✅ Yes, Save Item</Text>
                        </TouchableOpacity>

                        <TouchableOpacity style={styles.cancelButton} onPress={() => {
                            setShowConfirmation(false);
                            setConfirmationData(null);
                        }}>
                            <Text style={styles.cancelButtonText}>Cancel</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        padding: 20,
        paddingTop: 50,
    },
    // ... existing UI styles ...
    headerContainer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
    sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
    header: { fontSize: 24, fontWeight: 'bold' },
    comboButton: { backgroundColor: '#000', paddingHorizontal: 15, paddingVertical: 8, borderRadius: 20 },
    comboButtonText: { color: '#fff', fontWeight: 'bold', fontSize: 14 },
    createButton: { backgroundColor: '#7c3aed', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 15 },
    createButtonText: { color: '#fff', fontWeight: 'bold', fontSize: 12 },
    wardrobeList: { height: 120 },
    wardrobeItem: { marginRight: 15, borderRadius: 10, overflow: 'hidden', borderWidth: 1, borderColor: '#ddd' },
    wardrobeImage: { width: 100, height: 100 },
    divider: { height: 1, backgroundColor: '#eee', marginVertical: 20 },
    recItem: { flex: 1, margin: 5, backgroundColor: '#f9f9f9', borderRadius: 10, padding: 10, alignItems: 'center' },
    recImage: { width: '100%', height: 150, borderRadius: 10, marginBottom: 5 },
    recName: { fontSize: 14, fontWeight: '600' },
    recCategory: { fontSize: 10, color: '#666', marginBottom: 2 },
    recScore: { fontSize: 12, color: 'green' },
    placeholder: { textAlign: 'center', color: '#888', marginTop: 50 },
    iconButton: { backgroundColor: '#000', padding: 8, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },

    // Loading Styles
    loadingOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'center', alignItems: 'center' },
    loadingContainer: { backgroundColor: 'white', padding: 24, borderRadius: 16, alignItems: 'center', elevation: 5, width: '80%' },
    loadingText: { marginTop: 12, fontSize: 16, fontWeight: '600', color: '#333', textAlign: 'center' },
    progressBarContainer: { width: '100%', height: 8, backgroundColor: '#eee', borderRadius: 4, marginTop: 15, overflow: 'hidden' },
    progressBarFill: { height: '100%', backgroundColor: '#7c3aed' },
    progressText: { fontSize: 10, color: '#888', marginTop: 4 },

    // Confirmation Modal Styles
    modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.8)', justifyContent: 'center', alignItems: 'center' },
    confirmationContainer: { backgroundColor: 'white', padding: 20, borderRadius: 16, width: '90%', alignItems: 'center' },
    modalTitle: { fontSize: 20, fontWeight: 'bold', marginBottom: 15 },
    previewImage: { width: 120, height: 120, borderRadius: 10, marginBottom: 15, backgroundColor: '#f0f0f0' },
    modalSubtitle: { fontSize: 16, textAlign: 'center', marginBottom: 15 },
    categoryList: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 8 },
    categoryChip: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 20, backgroundColor: '#f0f0f0', borderWidth: 1, borderColor: '#ddd' },
    selectedCategoryChip: { backgroundColor: '#7c3aed', borderColor: '#7c3aed' },
    categoryChipText: { fontSize: 12, color: '#333' },
    selectedCategoryChipText: { color: 'white', fontWeight: 'bold' },
    confirmButton: { backgroundColor: '#000', paddingVertical: 12, paddingHorizontal: 30, borderRadius: 25, marginTop: 10, width: '100%', alignItems: 'center' },
    confirmButtonText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
    cancelButton: { marginTop: 10, padding: 10 },
    cancelButtonText: { color: '#666' },

});
