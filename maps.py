"""
A Yelp-powered Restaurant Recommendation Program
Unsupervised Learning & Supervised Learning
Machine Learning Project

"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

# Unsupervised Learning

def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location. If
    multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    distances = []
    for i in centroids:
        calculation = distance(location, i)
        distances.append(calculation)
    iloc = distances.index(min(distances))
    return centroids[iloc]

def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    groups = []
    for r in restaurants:
        close = find_closest(restaurant_location(r), centroids)
        groups.append((close, r))
    return group_by_first(groups)


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    lats = []
    longs = []
    for i in cluster:
        lats.append(restaurant_location(i)[0])
        longs.append(restaurant_location(i)[1])
    avg_lat = mean(lats)
    avg_long = mean(longs)
    return [avg_lat, avg_long]


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    prev_centroids, n = [], 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while prev_centroids != centroids and n < max_updates:
        prev_centroids = centroids
        clusters = group_by_centroid(restaurants, prev_centroids)
        centroids = []
        for cluster in clusters:
            centroid = find_centroid(cluster)
            centroids.append(centroid)
        n += 1
    return centroids

# Supervised Learning

def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_rating(review)
                       for review in user_reviews(user).values()}

    xvals = [feature_fn(r) for r in restaurants]
    yvals = [reviews_by_user[restaurant_name(r)] for r in restaurants]

    meanx = mean(xvals)
    meany = mean(yvals)

    sxx = sum([(x - meanx)**2 for x in xvals])
    syy = sum([(y - meany)**2 for y in yvals])
    sxy = sum([(x - meanx) * (y - meany) for x, y in zip(xvals, yvals)])

    b = (sxy / sxx)
    a = meany - (b * meanx)
    r_squared = (sxy**2) / (sxx * syy)

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    predictors = []
    r_vals = []
    for feature in feature_fns:
         r_vals.append(find_predictor(user, reviewed, feature)[1])
         predictors.append(find_predictor(user, reviewed, feature)[0])
    ind = r_vals.index(max(r_vals))
    return predictors[ind]


def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)

    predicted = []
    for r in restaurants:
        if r in reviewed:
            name = restaurant_name(r)
            rating = int(user_rating(user, name))
            predicted.append((name, rating))

        else:
            name = restaurant_name(r)
            rating = predictor(r)
            predicted.append((name, rating))

    return dict(predicted)

def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    return [r for r in restaurants if query in restaurant_categories(r)]


def feature_set():
    """Return a sequence of feature functions."""
    return [restaurant_mean_rating,
            restaurant_price,
            restaurant_num_ratings,
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]

""" Everything below is not my work """

@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
