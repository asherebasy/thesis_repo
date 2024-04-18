import numpy as np
import pandas as pd
from tqdm import tqdm

from DataReader.SDAPIDataReader import SDAPIDataReader
from DataReader.TrackingDataReader import TrackingDataReader
from config.rb_team_ids import rb_team_to_ids

tournament_calendars = {
    '2. Liga': {
        '20_21': '8fcwecynkexaf0d2nrlb853pw',
        '21_22': '9in40r1msgsry1et3rdf3dn9w',
        '22_23': '68w8zeyv7nezb0y0g7ijdaluc',
        '23_24': '21i3op95jwpzyo13gpo8f3gno'
    },
    'BL': {
        '20_21': '8o5hjwtoaw9q3ndqmrjnmuzqi',
        '21_22': '29p4usgcyyx23x1gjpiksz310',
        '22_23': 'b0r2dzuwepenhesbl5yrnhr10',
        '23_24': 'a5rfzm8qpy3ca8d8wxlkkdslw'
    },
    'CUP': {
        '20_21': 'aq8a62kn1olqqp8sbk1yerzhm',
        '21_22': '5oq32zzhhu8envjf7p83vx5p0',
        '22_23': '25rqzyxotfthkjyf01es2wxzo',
        '23_24': '9oxqvhc0x1mcc01hly1cc2rys'
    },
    'UCL': {
        '20_21': '4xs9rnedg5og3phf4eaxg5ylm',
        '21_22': '1sc9sn2keddyalfj2z0wy77ys',
        '22_23': '8t8ofk94zy6ksx1spnfecvpck',
        '23_24': 'eaf2z13av1rs0jbwnnxfapdec'
    }
}
liefering_id = rb_team_to_ids['API']['fcl']
rbs_id = rb_team_to_ids['API']['profis']
faulty_fixtures = ['dy24eqzqlktqccfmzpc8ay7tg']
grid_kwargs = {'grid_height': 1, 'endnote_height': 0, 'bottom': 0, 'endnote_space': 0, 'title_space': 0,
               'title_height': 0}
log_column_names = ['match_id', 'match_description', 'match_date', 'team', 'opponent', 'seq_id',
                    'seq_length']
tr_download_directory = '/Users/ammar.elsherebasy/Desktop/LIVE/downloads'
viz_dir = '/Users/ammar.elsherebasy/Desktop/Thesis/Generated Images/'

outletAuthKey = '154u35hcigs30149l08bq1rjof'
secretKey = '1u7qwexbfdjvy114ai14a3oiyj'


def get_fixture_list_from_all_calendars() -> pd.DataFrame:
    """
    Retrieves fixture lists from all available tournament calendars.

    This function retrieves fixture lists from all available tournament calendars and filters out played matches
    with sufficient coverage level. Additionally, it filters fixture lists based on specific competitions such as
    '2. Liga', 'CUP', and 'UCL'. It concatenates all available fixtures and returns a DataFrame containing the
    filtered fixture information.

    Returns:
        pd.DataFrame: DataFrame containing fixture information including match ID, home team ID, away team ID, and
        other relevant details for each fixture.
    """
    caller = SDAPIDataReader(outlet=outletAuthKey, secret=secretKey, df_option=True)

    df = pd.DataFrame.from_dict(tournament_calendars)
    tournaments_df = pd.melt(df, ignore_index=False).reset_index()
    tournaments_df.rename(columns={'index': 'season', 'variable': 'competition', 'value': 'tcl_id'},
                          inplace=True)

    available_fixtures = pd.DataFrame()
    for ix, row in tournaments_df.iterrows():
        tournamentCalendarUuid = row['tcl_id']

        queryParameters = {
            "tmcl": [tournamentCalendarUuid],
            "_pgSz": ["900"]
        }
        fixture_list = caller.read_MA1(queryParameters=queryParameters, detailed=False)
        fixture_list = fixture_list[
            (fixture_list['liveData.matchDetails.matchStatus'] == 'Played') &
            (fixture_list['matchInfo.coverageLevel'].astype(float) >= 13)
            ]
        fixture_list['sp_home_team_id'] = fixture_list['matchInfo.contestant'].apply(lambda x: x[0]['id'])
        fixture_list['sp_away_team_id'] = fixture_list['matchInfo.contestant'].apply(lambda x: x[1]['id'])

        if row.competition == '2. Liga':
            fixture_list = fixture_list[
                (fixture_list.sp_home_team_id == liefering_id) | (fixture_list.sp_away_team_id == liefering_id)
                ]

        if row.competition in ['CUP', 'UCL']:
            fixture_list = fixture_list[
                (fixture_list.sp_home_team_id == rbs_id) | (fixture_list.sp_away_team_id == rbs_id)
                ]

        available_fixtures = pd.concat([available_fixtures, fixture_list])

    available_fixtures = available_fixtures[~available_fixtures['matchInfo.id'].isin(faulty_fixtures)].reset_index(
        drop=True)
    return available_fixtures


def read_log_file(log_file: str) -> pd.DataFrame:
    try:
        log_df = pd.read_csv(log_file)
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=log_column_names)

    return log_df


def write_to_log_file(log_df: pd.DataFrame,
                      filename: str = 'log.csv') -> None:
    log_df.to_csv(filename)


def generate_representations(fixture_list: pd.DataFrame,
                             sprint_threshold: float = 8.3,
                             sprint_window: int = 1,
                             log_file: str = None):
    if log_file is None:
        log_file = 'log.csv'
    log_df = read_log_file(log_file=log_file)
    read_fixtures = log_df.match_id.unique()

    for _, row in tqdm(fixture_list.iterrows(), total=fixture_list.shape[0]):
        fx_id = row['matchInfo.id']
        if fx_id in read_fixtures:
            continue
        else:
            match_description = row['matchInfo.description']
            match_date = row['matchInfo.localDate']

        try:
            tr_data_Reader = TrackingDataReader(
                outlet=outletAuthKey,
                secret=secretKey,
                fixtureUuid=fx_id,
                download_directory=tr_download_directory,
                fps=10
                )
        except ValueError:
            log_row = {'match_id': fx_id, 'match_description': match_description, 'match_date': match_date,
                       'team': 'VALUE ERROR', 'opponent': 'VALUE ERROR', 'seq_id': 'VALUE ERROR',
                       'seq_length': 'VALUE ERROR'}
            log_df.loc[len(log_df)] = log_row
            write_to_log_file(log_df=log_df, filename=log_file)
            continue
        except AssertionError:
            log_row = {'match_id': fx_id, 'match_description': match_description, 'match_date': match_date,
                       'team': 'ASSERTION ERROR', 'opponent': 'ASSERTION ERROR', 'seq_id': 'ASSERTION ERROR',
                       'seq_length': 'ASSERTION ERROR'}
            log_df.loc[len(log_df)] = log_row
            write_to_log_file(log_df=log_df, filename=log_file)
            continue
        except KeyError:
            log_row = {'match_id': fx_id, 'match_description': match_description, 'match_date': match_date,
                       'team': 'KEY ERROR', 'opponent': 'KEY ERROR', 'seq_id': 'KEY ERROR',
                       'seq_length': 'KEY ERROR'}
            log_df.loc[len(log_df)] = log_row
            write_to_log_file(log_df=log_df, filename=log_file)
            continue

        teams = tr_data_Reader.unique_teams
        sprints_matrix = tr_data_Reader.generate_sprints_matrix(sprint_threshold=sprint_threshold,
                                                                sprint_window=sprint_window)
        for team in teams:
            opponent_id = teams[teams != team][0]

            team_poss_tr = sprints_matrix[sprints_matrix.ball_besitz == team]
            team_poss_tr = team_poss_tr[
                (team_poss_tr[tr_data_Reader.squads[team]['']] == 1).any(axis=1)
            ]  # sprints by players of team
            team_poss_tr.dropna(how="all", axis=0, inplace=True)
            sequences_with_sprints = team_poss_tr.groupby('ball_besitz_sequence').agg(
                {player: sum for player in tr_data_Reader.squads[team]['']}
            ) >= sprint_window
            sequences_with_sprints = sequences_with_sprints.sum(axis=1)
            sequences_with_sprints = sequences_with_sprints[sequences_with_sprints > 0].index
            tr_data_Reader.lolatos(
                team_of_interest=team,
                ball_besitz_sequences=sequences_with_sprints,
                save_dir=viz_dir,
                save_dpi=32,
                plot_title=False,
                figheight=10,
                grid_kwargs=grid_kwargs
            )

            no_sequences = len(sequences_with_sprints)
            seq_length = tr_data_Reader.tr[
                tr_data_Reader.tr.ball_besitz_sequence.isin(sequences_with_sprints)
            ].groupby('ball_besitz_sequence').agg(
                {'seconds': np.ptp}
            ).values

            merged_log_rows = pd.DataFrame(
                dict(
                    zip(log_column_names,
                        [[fx_id] * no_sequences, [match_description] * no_sequences, [match_date] * no_sequences,
                         [team] * no_sequences,
                         [opponent_id] * no_sequences, sequences_with_sprints.tolist(), (seq_length / 1000).T[0]]
                        )
                )
            )
            log_df = pd.concat([log_df, merged_log_rows], axis=0, ignore_index=True).reset_index(drop=True)
            write_to_log_file(log_df=log_df, filename=log_file)


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    fixtures = get_fixture_list_from_all_calendars()
    generate_representations(
        fixture_list=fixtures,
        log_file='/Users/ammar.elsherebasy/Desktop/Thesis/generated_games_log.csv'
        )
