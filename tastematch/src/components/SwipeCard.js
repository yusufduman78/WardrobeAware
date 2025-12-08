import React from 'react';
import { StyleSheet, Text, View, Image, Dimensions } from 'react-native';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    runOnJS,
} from 'react-native-reanimated';
import FontAwesome from '@expo/vector-icons/FontAwesome';

const { width, height } = Dimensions.get('window');

const SWIPE_THRESHOLD = width * 0.25;
const VERTICAL_SWIPE_THRESHOLD = height * 0.1;

const SwipeCard = ({ item, onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown }) => {
    const translateX = useSharedValue(0);
    const translateY = useSharedValue(0);

    const panGesture = Gesture.Pan()
        .onUpdate((event) => {
            translateX.value = event.translationX;
            translateY.value = event.translationY;
        })
        .onEnd(() => {
            // Horizontal Swipes (Like/Dislike)
            if (Math.abs(translateX.value) > Math.abs(translateY.value)) {
                if (translateX.value > SWIPE_THRESHOLD) {
                    translateX.value = withSpring(width + 100);
                    runOnJS(onSwipeRight)();
                } else if (translateX.value < -SWIPE_THRESHOLD) {
                    translateX.value = withSpring(-width - 100);
                    runOnJS(onSwipeLeft)();
                } else {
                    translateX.value = withSpring(0);
                    translateY.value = withSpring(0);
                }
            }
            // Vertical Swipes (Superlike/Watchlist) - Only if NOT a recommendation
            else if (!item.is_recommendation) {
                if (translateY.value < -VERTICAL_SWIPE_THRESHOLD) {
                    // Swipe Up (Superlike)
                    translateY.value = withSpring(-height - 100);
                    if (onSwipeUp) runOnJS(onSwipeUp)();
                } else if (translateY.value > VERTICAL_SWIPE_THRESHOLD) {
                    // Swipe Down (Watchlist)
                    translateY.value = withSpring(height + 100);
                    if (onSwipeDown) runOnJS(onSwipeDown)();
                } else {
                    translateX.value = withSpring(0);
                    translateY.value = withSpring(0);
                }
            } else {
                // If it IS a recommendation, reset vertical position (disable vertical swipe)
                translateX.value = withSpring(0);
                translateY.value = withSpring(0);
            }
        });

    const cardStyle = useAnimatedStyle(() => {
        const rotate = `${translateX.value / 20}deg`;
        return {
            transform: [
                { translateX: translateX.value },
                { translateY: translateY.value },
                { rotate: rotate },
            ],
        };
    });

    const meta = item.metadata_content || {};
    const rating = item.vote_average || meta.vote_average || meta.rating;
    const genres = item.genres || meta.genres || 'Unknown';
    const description = item.overview || meta.description || meta.overview || 'No description available.';

    return (
        <GestureDetector gesture={panGesture}>
            <Animated.View style={[styles.card, cardStyle]}>
                <Image source={{ uri: item.image_url }} style={styles.image} />

                {/* Match Badge */}
                {/* Match Badges */}
                {item.match_type === 'perfect' && (
                    <View style={[styles.matchBadge, styles.perfectMatchBadge]}>
                        <Text style={styles.matchText}>IT'S A MATCH! 🎯</Text>
                    </View>
                )}

                {item.match_type === 'reverse' && (
                    <View style={[styles.matchBadge, styles.reverseMatchBadge]}>
                        <Text style={styles.matchText}>DEFINITELY NOT YOUR TASTE 🤪</Text>
                        <Text style={styles.subMatchText}>BUT TRY IT!</Text>
                    </View>
                )}

                <View style={styles.infoContainer}>
                    <Text style={styles.title}>{item.title}</Text>
                    <View style={styles.metaRow}>
                        <Text style={styles.type}>{item.type.toUpperCase()}</Text>
                        {rating && (
                            <View style={styles.ratingContainer}>
                                <FontAwesome name="star" size={14} color="#FFD700" />
                                <Text style={styles.ratingText}>{Number(rating).toFixed(1)}</Text>
                            </View>
                        )}
                    </View>
                    <Text style={styles.genres}>{genres}</Text>
                    <Text numberOfLines={3} style={styles.description}>
                        {description}
                    </Text>
                </View>
            </Animated.View>
        </GestureDetector>
    );
};

const styles = StyleSheet.create({
    card: {
        width: width * 0.9,
        height: height * 0.7,
        backgroundColor: 'white',
        borderRadius: 20,
        shadowColor: '#000',
        shadowOffset: {
            width: 0,
            height: 2,
        },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
        elevation: 5,
        position: 'absolute',
        alignSelf: 'center',
        top: 50,
    },
    image: {
        width: '100%',
        height: '65%',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        resizeMode: 'cover',
    },
    infoContainer: {
        padding: 20,
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 5,
    },
    metaRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 5,
    },
    type: {
        fontSize: 12,
        fontWeight: 'bold',
        color: '#888',
    },
    ratingContainer: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    ratingText: {
        marginLeft: 5,
        fontSize: 14,
        fontWeight: 'bold',
        color: '#333',
    },
    genres: {
        fontSize: 12,
        color: '#0a7ea4',
        marginBottom: 10,
        fontStyle: 'italic',
    },
    description: {
        fontSize: 14,
        color: '#444',
    },
    matchBadge: {
        position: 'absolute',
        top: 20,
        right: 20,
        backgroundColor: '#FFD700',
        paddingHorizontal: 15,
        paddingVertical: 8,
        borderRadius: 20,
        transform: [{ rotate: '15deg' }],
        borderWidth: 2,
        borderColor: 'white',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 3,
        elevation: 5,
        zIndex: 10,
    },
    matchText: {
        color: '#fff',
        fontWeight: '900',
        fontSize: 14,
        textAlign: 'center',
    },
    subMatchText: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 10,
        textAlign: 'center',
    },
    perfectMatchBadge: {
        backgroundColor: '#FFD700', // Gold
        borderColor: '#fff',
    },
    reverseMatchBadge: {
        backgroundColor: '#8A2BE2', // BlueViolet
        borderColor: '#fff',
        transform: [{ rotate: '-15deg' }], // Rotate opposite way
    },
});

export default SwipeCard;
