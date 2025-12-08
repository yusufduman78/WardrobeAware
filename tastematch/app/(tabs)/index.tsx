import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Image, Dimensions, ActivityIndicator, TouchableOpacity } from 'react-native';
import { GestureHandlerRootView, GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, { useAnimatedStyle, useSharedValue, withSpring, runOnJS } from 'react-native-reanimated';
import axios from 'axios';
import Config from '../../constants/Config';
import { Ionicons } from '@expo/vector-icons';

const SCREEN_WIDTH = Dimensions.get('window').width;
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.25;

interface Item {
  id: string;
  name: string;
  price: number;
  image_url: string;
  category_id: string;
}

export default function SwipeScreen() {
  const [items, setItems] = useState<Item[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const contextX = useSharedValue(0);
  const contextY = useSharedValue(0);

  useEffect(() => {
    fetchItems();
  }, []);

  const fetchItems = async () => {
    try {
      setLoading(true);
      setError(false);
      // Fetch 20 items at a time
      const response = await axios.get(`${Config.API_URL}/feed?limit=20`);
      if (response.data.length === 0) {
        setLoading(false);
        return;
      }

      setItems(prev => {
        // Avoid duplicates if any
        const newItems = response.data.filter((newItem: Item) => !prev.some(existing => existing.id === newItem.id));
        return [...prev, ...newItems];
      });
      setLoading(false);
    } catch (error) {
      console.error("Error fetching items:", error);
      setLoading(false);
      setError(true);
    }
  };

  const onSwipeComplete = (direction: 'left' | 'right' | 'up' | 'down') => {
    const item = items[currentIndex];
    const actionMap = {
      left: 'dislike',
      right: 'like',
      up: 'superlike',
      down: 'save'
    };

    // Send action to backend
    axios.post(`${Config.API_URL}/swipe`, {
      user_id: 'test_user', // Mock user ID
      item_id: item.id,
      action: actionMap[direction]
    }).catch(err => console.error("Swipe error:", err));

    setCurrentIndex(prev => prev + 1);
    translateX.value = 0;
    translateY.value = 0;

    // Fetch more if running low (less than 5 items remaining)
    if (items.length - currentIndex < 5) {
      fetchItems();
    }
  };

  const panGesture = Gesture.Pan()
    .onStart(() => {
      contextX.value = translateX.value;
      contextY.value = translateY.value;
    })
    .onUpdate((event) => {
      translateX.value = contextX.value + event.translationX;
      translateY.value = contextY.value + event.translationY;
    })
    .onEnd((event) => {
      if (Math.abs(event.translationX) > SWIPE_THRESHOLD) {
        // Swipe Left or Right
        translateX.value = withSpring(Math.sign(event.translationX) * SCREEN_WIDTH * 1.5);
        runOnJS(onSwipeComplete)(event.translationX > 0 ? 'right' : 'left');
      } else if (event.translationY < -SWIPE_THRESHOLD) {
        // Swipe Up (Superlike)
        translateY.value = withSpring(-SCREEN_WIDTH * 1.5);
        runOnJS(onSwipeComplete)('up');
      } else if (event.translationY > SWIPE_THRESHOLD) {
        // Swipe Down (Save)
        translateY.value = withSpring(SCREEN_WIDTH * 1.5);
        runOnJS(onSwipeComplete)('down');
      } else {
        translateX.value = withSpring(0);
        translateY.value = withSpring(0);
      }
    });

  const cardStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
        { rotate: `${translateX.value / 10}deg` }
      ],
    };
  });

  if (loading && items.length === 0) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={{ marginTop: 10 }}>Finding styles for you...</Text>
      </View>
    );
  }

  if (error && items.length === 0) {
    return (
      <View style={styles.container}>
        <Text>Oops! Could not load items.</Text>
        <TouchableOpacity onPress={fetchItems} style={styles.retryButton}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (currentIndex >= items.length) {
    return (
      <View style={styles.container}>
        <Text>No more items!</Text>
        <TouchableOpacity onPress={fetchItems} style={styles.retryButton}>
          <Text style={styles.retryText}>Refresh Feed</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const currentItem = items[currentIndex];
  const nextItem = items[currentIndex + 1];

  return (
    <GestureHandlerRootView style={styles.container}>
      {nextItem && (
        <View style={[styles.card, styles.nextCard]} key={nextItem.id}>
          <Image
            source={{ uri: nextItem.image_url }}
            style={styles.image}
            resizeMode="cover"
            // Force reload if URL changes or component remounts
            key={nextItem.image_url}
          />
          <View style={styles.infoContainer}>
            <Text style={styles.name}>{nextItem.name}</Text>
            <Text style={styles.price}>${nextItem.price}</Text>
          </View>
        </View>
      )}

      <GestureDetector gesture={panGesture}>
        <Animated.View style={[styles.card, cardStyle]} key={currentItem.id}>
          <Image
            source={{ uri: currentItem.image_url }}
            style={styles.image}
            resizeMode="cover"
            key={currentItem.image_url}
          />
          <View style={styles.infoContainer}>
            <Text style={styles.name}>{currentItem.name}</Text>
            <Text style={styles.price}>${currentItem.price}</Text>
          </View>

          {/* Overlay Labels (Optional) */}
        </Animated.View>
      </GestureDetector>

      <View style={styles.controls}>
        <Ionicons name="close-circle" size={60} color="red" />
        <Ionicons name="heart-circle" size={60} color="green" />
      </View>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    width: SCREEN_WIDTH * 0.9,
    height: SCREEN_WIDTH * 1.3,
    backgroundColor: 'white',
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 5,
    position: 'absolute',
    overflow: 'hidden'
  },
  nextCard: {
    transform: [{ scale: 0.95 }],
    zIndex: -1,
  },
  image: {
    width: '100%',
    height: '80%',
  },
  infoContainer: {
    padding: 15,
  },
  name: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  price: {
    fontSize: 16,
    color: '#888',
  },
  controls: {
    position: 'absolute',
    bottom: 50,
    flexDirection: 'row',
    width: '100%',
    justifyContent: 'space-evenly',
    zIndex: -1
  },
  retryButton: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#007AFF',
    borderRadius: 10
  },
  retryText: {
    color: 'white',
    fontWeight: 'bold'
  }
});
