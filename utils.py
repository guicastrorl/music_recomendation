import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import random


# Functions
## General Functions
def calculate_flags(dataframe):
    max_id_date = dataframe['id_date'].max()
    
    dataframe['ultimos_03_meses_flag'] = dataframe.apply(lambda row: 1 if max_id_date - timedelta(days=90) <= row['id_date'] <= max_id_date else 0, axis=1)
    dataframe['ultimos_06_meses_flag'] = dataframe.apply(lambda row: 1 if max_id_date - timedelta(days=180) <= row['id_date'] <= max_id_date else 0, axis=1)
    dataframe['ultimos_09_meses_flag'] = dataframe.apply(lambda row: 1 if max_id_date - timedelta(days=270) <= row['id_date'] <= max_id_date else 0, axis=1)
    dataframe['ultimos_12_meses_flag'] = dataframe.apply(lambda row: 1 if max_id_date - timedelta(days=365) <= row['id_date'] <= max_id_date else 0, axis=1)
    
    return dataframe


def calculate_statistics(dataframe):
    statistics = {
        'VL_TOT_PLAYS': dataframe['plays'].sum(),
        'VL_MED_PLAYS': dataframe['plays'].mean(),
        'VL_MAX_PLAYS': dataframe['plays'].max(),
        'VL_MIN_PLAYS': dataframe['plays'].min()
    }
    
    for period in ['ultimos_03_meses_flag', 'ultimos_06_meses_flag', 'ultimos_09_meses_flag', 'ultimos_12_meses_flag']:
        filtered_plays = dataframe[dataframe[period] == 1]['plays']
        statistics[f'VL_TOT_U{period[8:10].upper()}M_PLAYS'] = round(filtered_plays.sum(), 2)
        statistics[f'VL_MED_U{period[8:10].upper()}M_PLAYS'] = round(filtered_plays.mean(), 2)
        statistics[f'VL_MAX_U{period[8:10].upper()}M_PLAYS'] = round(filtered_plays.max(), 2)
        statistics[f'VL_MIN_U{period[8:10].upper()}M_PLAYS'] = round(filtered_plays.min(), 2)
    
    return pd.Series(statistics)


def calculate_ratios(dataframe):
    ratios = {
        'VL_RAZ_TOT_U03M_U06M_PLAYS': round(dataframe['VL_TOT_U03M_PLAYS'].sum() / dataframe['VL_TOT_U06M_PLAYS'].sum(), 2),
        'VL_RAZ_TOT_U03M_U09M_PLAYS': round(dataframe['VL_TOT_U03M_PLAYS'].sum() / dataframe['VL_TOT_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_TOT_U03M_U12M_PLAYS': round(dataframe['VL_TOT_U03M_PLAYS'].sum() / dataframe['VL_TOT_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MED_U03M_U06M_PLAYS': round(dataframe['VL_MED_U03M_PLAYS'].sum() / dataframe['VL_MED_U06M_PLAYS'].sum(), 2),
        'VL_RAZ_MED_U03M_U09M_PLAYS': round(dataframe['VL_MED_U03M_PLAYS'].sum() / dataframe['VL_MED_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_MED_U03M_U12M_PLAYS': round(dataframe['VL_MED_U03M_PLAYS'].sum() / dataframe['VL_MED_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MAX_U03M_U06M_PLAYS': round(dataframe['VL_MAX_U03M_PLAYS'].sum() / dataframe['VL_MAX_U06M_PLAYS'].sum(), 2),
        'VL_RAZ_MAX_U03M_U09M_PLAYS': round(dataframe['VL_MAX_U03M_PLAYS'].sum() / dataframe['VL_MAX_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_MAX_U03M_U12M_PLAYS': round(dataframe['VL_MAX_U03M_PLAYS'].sum() / dataframe['VL_MAX_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MIN_U03M_U06M_PLAYS': round(dataframe['VL_MIN_U03M_PLAYS'].sum() / dataframe['VL_MIN_U06M_PLAYS'].sum(), 2),
        'VL_RAZ_MIN_U03M_U09M_PLAYS': round(dataframe['VL_MIN_U03M_PLAYS'].sum() / dataframe['VL_MIN_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_MIN_U03M_U12M_PLAYS': round(dataframe['VL_MIN_U03M_PLAYS'].sum() / dataframe['VL_MIN_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_TOT_U06M_U09M_PLAYS': round(dataframe['VL_TOT_U06M_PLAYS'].sum() / dataframe['VL_TOT_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_TOT_U06M_U12M_PLAYS': round(dataframe['VL_TOT_U06M_PLAYS'].sum() / dataframe['VL_TOT_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MED_U06M_U09M_PLAYS': round(dataframe['VL_MED_U06M_PLAYS'].sum() / dataframe['VL_MED_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_MED_U06M_U12M_PLAYS': round(dataframe['VL_MED_U06M_PLAYS'].sum() / dataframe['VL_MED_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MAX_U06M_U09M_PLAYS': round(dataframe['VL_MAX_U06M_PLAYS'].sum() / dataframe['VL_MAX_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_MAX_U06M_U12M_PLAYS': round(dataframe['VL_MAX_U06M_PLAYS'].sum() / dataframe['VL_MAX_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MIN_U06M_U09M_PLAYS': round(dataframe['VL_MIN_U06M_PLAYS'].sum() / dataframe['VL_MIN_U09M_PLAYS'].sum(), 2),
        'VL_RAZ_MIN_U06M_U12M_PLAYS': round(dataframe['VL_MIN_U06M_PLAYS'].sum() / dataframe['VL_MIN_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_TOT_U09M_U12M_PLAYS': round(dataframe['VL_TOT_U09M_PLAYS'].sum() / dataframe['VL_TOT_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MED_U09M_U12M_PLAYS': round(dataframe['VL_MED_U09M_PLAYS'].sum() / dataframe['VL_MED_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MAX_U09M_U12M_PLAYS': round(dataframe['VL_MAX_U09M_PLAYS'].sum() / dataframe['VL_MAX_U12M_PLAYS'].sum(), 2),
        'VL_RAZ_MIN_U09M_U12M_PLAYS': round(dataframe['VL_MIN_U09M_PLAYS'].sum() / dataframe['VL_MIN_U12M_PLAYS'].sum(), 2)
    }

    return pd.Series(ratios)


def collect_top_id_tracks(dataframe):
    # Sort the DataFrame by 'plays' in descending order
    dataframe_sorted = dataframe.sort_values(by='plays', ascending=False)
    
    # Group by 'user_id' and collect the top 10 'id_tracks' for each group
    top_10_id_tracks = dataframe_sorted.groupby('user_id')['id_tracks'].apply(lambda x: x.head(10).astype(str).tolist()).reset_index()
    
    # Rename the column to 'top_10_id_tracks'
    top_10_id_tracks = top_10_id_tracks.rename(columns={'id_tracks': 'top_10_id_tracks'})
    
    return top_10_id_tracks
def generate_metadata(dataframe):
    """
    Gera um dataframe contendo metadados das colunas do dataframe fornecido.

    :param dataframe: DataFrame para o qual os metadados serão gerados.
    :return: DataFrame contendo metadados.
    """

    # Coleta de metadados básicos
    metadata = pd.DataFrame({
        'nome_variavel': dataframe.columns,
        'tipo': dataframe.dtypes,
        'quantidade_nulos': dataframe.isnull().sum(),
        'percentual_nulos': round((dataframe.isnull().sum() / len(dataframe))* 100,2),
        'cardinalidade': dataframe.nunique(),
    })
    metadata = metadata.sort_values(by='tipo')
    metadata = metadata.reset_index(drop=True)

    return metadata
## Recommender Class and
class SpotifyRecommender():
    def __init__(self, recommendation_data) -> None:
        self.recommendation_data = recommendation_data


    def min_manhattan_distance(self, row, matrix):
        distances = np.sum(np.abs(matrix - row), axis=1)
        return np.min(distances)


    def jaccard_similarity(self, matrix1, matrix2):
        intersection = len(set(matrix1.flatten()).intersection(set(matrix2.flatten())))
        union = len(set(matrix1.flatten()).union(set(matrix2.flatten())))
        return (intersection / union) * 100


    def spotify_recommendations(self, id_track, amount=1, verbose=False):
        try:
            # Filter tracks by the given id_track
            id_track_features = self.recommendation_data[self.recommendation_data.id_tracks.isin(id_track)]
            non_id_track_data = self.recommendation_data[~self.recommendation_data.id_tracks.isin(id_track)]

            # Extract relevant columns for comparison (excluding specific columns by index)
            a = id_track_features.iloc[:, [id_column for id_column in np.arange(len(self.recommendation_data.columns)) if not id_column in [0, 1, 46]]]
            b = non_id_track_data.iloc[:, [id_column for id_column in np.arange(len(self.recommendation_data.columns)) if not id_column in [0, 1, 46]]]

            matrix1 = a.values
            matrix2 = b.values

            # Calculate Manhattan distances between id_track_features and other tracks
            non_id_track_data['distance'] = np.apply_along_axis(self.min_manhattan_distance, 1, matrix2, matrix=matrix1)
            non_id_track_data = non_id_track_data.sort_values('distance')

            if verbose:
                # Achatar as matrizes em vetores
                vector1 = matrix1.flatten()
                vector2 = matrix2.flatten()

                # Adicionar zeros para tornar os vetores do mesmo tamanho
                max_len = max(len(vector1), len(vector2))
                vector1 = np.pad(vector1, (0, max_len - len(vector1)), 'constant')
                vector2 = np.pad(vector2, (0, max_len - len(vector2)), 'constant')

                # Normalizar os vetores (importante para o cálculo de similaridade do cosseno)
                vector1 = normalize(vector1.reshape(1, -1))
                vector2 = normalize(vector2.reshape(1, -1))

                # Calcular a similaridade do cosseno entre os vetores
                cosine_similarity_value = cosine_similarity(vector1, vector2)

                print(f'KPI: {-cosine_similarity_value[0][0] * 1000:.2f}%')

            # Select columns for return and limit the results to the specified amount
            columns_to_return = ['id_tracks', 'id_artist', 'distance']

            return non_id_track_data[columns_to_return][:amount]

        except Exception as e:
            print(e)
            print('It is not possible to recommend for this track id.')

## Artefacts
# data transactional function
def create_df_modeling(start_date, end_date, low_user, high_user, dau, low_track, high_track, min_plays, max_plays, min_n_tracks, max_n_tracks):
    # creates a range of dates
    date_range = pd.date_range(start_date, end_date)

    # creates empty array for the final user array
    user_array_acum = []

    # creates empty array for the final id_date array
    date_array_acum = []

    # creates empty array for the final track array
    tracks_array_acum = []

    # creates empty array for the final plays array
    plays_array_acum = []

    # minimum number of random listeners per day (dau)
    length_user = 1 + int((high_user - low_user) * dau)

    # for loop to random create transactional data
    for current_date in date_range:
        #
        user_array = np.random.randint(low_user, high_user, size = length_user)
        user_array_col = np.reshape(user_array, (len(user_array), 1))
        user_array_acum.append(user_array_col)
        #
        date_array = current_date.strftime("%Y-%m-%d")
        date_array = np.full(len(user_array), current_date)
        date_array_col = np.reshape(date_array, (len(date_array), 1))
        date_array_acum.append(date_array_col)

    # for loop that ranges from the min id_user to the max id_user
    for k in range(0,len(np.concatenate(user_array_acum))):
        # defines randomly the maximum range of tracks a listener can listen in 1 single day
        length_size = random.randint(min_n_tracks, max_n_tracks)
        # selects at random an array of id_track for the specific listener
        tracks_array = np.random.randint(low_track, high_track, size = length_size)
        tracks_array_acum.append(tracks_array)
        # select at random the average of plays given by each user on a giving day
        plays_array = np.random.randint(min_plays, max_plays, size = length_size)
        plays_array_acum.append(plays_array)

    # concatenates the arrays of date, id_user and plays on the final dataframe
    date_array_final = np.concatenate(date_array_acum, axis = 0)
    user_array_final = np.concatenate(user_array_acum, axis = 0)

    # creates the final transactional dataframe with id_users and number of plays for each day
    df = pd.DataFrame({
        'id_date': date_array_final.reshape(len(date_array_final)),
        'user_id': user_array_final.reshape(len(user_array_final)),
        'id_tracks': tracks_array_acum,
        'plays': plays_array_acum
    })

    # explodes the final dataframe to create a vertical transactional table
    df = df.explode(['id_tracks', 'plays']).reset_index(drop=True)

    #
    return df

# data dimension function
def create_dim_content(n_tracks, min_tracks, max_tracks, n_artists, n_genres):
    # creates an initial list of id_tracks
    id_track_list = [i for i in range(1,n_tracks+1)]
    # creates an initial list of id_artists
    id_artist_list = [i for i in range(1, n_artists+1)]
    # creates an initial list of id_genres
    id_genres = [i for i in range(1,n_genres)]
    # creates array with the specific size of tracks by artists to be distributed amongst all artists
    artist_array_sizes = np.random.randint(min_tracks, max_tracks, size = n_artists).tolist()
    # Create a copy of the list
    list_copy = id_track_list.copy()
    # initiates an empty array to store results of for loop append (arrays of id_tracks)
    sampled_arrays_acum = []
    # initiates an empty array to store results of for loop append (arrays of id_genre)
    sampled_arrays_genre_acum = []
    # for loop to interate and create the dimensional data for id_artists and id_tracks
    for size in artist_array_sizes:
        if size <= len(list_copy):
            sampled_arrays = random.sample(list_copy, size)
            sampled_arrays_genre = random.choices(id_genres, k = size)
            sampled_arrays_acum.append(sampled_arrays)
            sampled_arrays_genre_acum.append(sampled_arrays_genre)
            list_copy = [item for item in list_copy if item not in sampled_arrays]
    # creates the final dimensional dataframe
    df = pd.DataFrame({'id_artist':id_artist_list,
                       'id_tracks':sampled_arrays_acum,
                       'id_genre':sampled_arrays_genre_acum
                      })
    # explodes the final dataframe to create a vertical dimensional table
    df = df.explode(['id_tracks','id_genre']).reset_index(drop = True)
    #
    return df

# data features function
def creates_features(n_features,n_artists,std_max, min_mean, max_mean):
    # creates an array with standard deviation of a uniform distribution
    std_array = [random.uniform(0, std_max) for i in range(0,n_features)]
    # creates an array with mean values of a uniform distribution
    mean_array = [random.randint(min_mean, max_mean) for i in range(0,n_features)]
    # creates an empty array with the number of rows as the same of the length of the quantity of artists
    result = np.empty((n_artists, 0))
    for i in range(0,n_features):
        feature_i = np.random.normal(mean_array[i], std_array[i] * mean_array[i], n_artists)
        feature_i = feature_i.astype(int)
        result = np.column_stack((result, feature_i))
        result = result.astype(int)
    # creates the list of id_artist
    id_artist = [i for i in range(1,n_artists+1)]
    # fills values on the result array
    result = np.column_stack((id_artist,result))
    # creates a list with only one element
    artist_col_names = ['id_artist']
    # creates a list with the names of the features that will be used on the final dataframe
    feat_col_names = ['Feature'+str(i) for i in range(1,n_features+1)]
    # adds two lists (elements will be use as the header of the final dataframe)
    col_names = artist_col_names + feat_col_names
    # final dataframe with features by artist
    df = pd.DataFrame(result, columns = col_names)
    return df