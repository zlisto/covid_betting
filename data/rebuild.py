from utils import *

__author__ = 'kq'


class Odds:

    @classmethod
    def american_to_decimal(cls, odds: float) -> float:
        return (1 + odds / 100) if odds > 0 else 1 - (100 / odds)

    @classmethod
    def decimal_to_probability(cls, odds: float) -> float:
        return 1 / odds


class Performance:

    @classmethod
    def win_loss(cls, data):
        win_loss = data.win_underdog.value_counts() / data.win_underdog.value_counts().sum()
        win_loss.index = win_loss.index.map({-1: 'loss', 1: 'win'})
        return win_loss

    @classmethod
    def profit(cls, data):
        return data.underdog_profit.sum() / len(data)


class Summary:

    @classmethod
    def quantile(cls, data: pd.DataFrame, quantiles: int = 10) -> pd.DataFrame:
        results = []
        data['q_ml'] = pd.qcut(data.ML_underdog, [j / quantiles for j in range(quantiles)],
                               [str(j) for j in range(quantiles - 1)])
        for func in [Performance.win_loss, Performance.profit]:
            results.append(data.groupby(data.q_ml).apply(func))
        results = pd.concat(results, axis=1)
        results.columns = ['loss', 'win', 'returns']
        return results.style.background_gradient(cmap='RdYlGn')

    @classmethod
    def monthly(cls, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for func in [Performance.win_loss, Performance.profit]:
            results.append(data.groupby(data.date.dt.month).apply(func))
        results = pd.concat(results, axis=1)
        results.columns = ['loss', 'win', 'returns']
        return results.style.background_gradient(cmap='RdYlGn')

    @classmethod
    def yearly(cls, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for func in [Performance.win_loss, Performance.profit]:
            results.append(data.groupby(data.date.dt.year).apply(func))
        results = pd.concat(results, axis=1)
        results.columns = ['loss', 'win', 'returns']
        return results.style.background_gradient(cmap='RdYlGn')

    @classmethod
    def season(cls, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for func in [Performance.win_loss, Performance.profit]:
            results.append(data.groupby(data.season).apply(func))
        results = pd.concat(results, axis=1)
        results.columns = ['loss', 'win', 'returns']
        return results.style.background_gradient(cmap='RdYlGn')


class Data:

    @classmethod
    def format_games(cls, year: Union[int, str], league: str = 'nba'):
        start, end = str(year), str(int(year) + 1)
        target = SOURCE.format(league.lower(), league.lower(), start, end[2:])
        if league == 'mlb':
            target = SOURCE.replace('-{}', '').format(league.lower(), league.lower(), start)
        elif league == 'ncaabasketball':
            target = SOURCE.replace('%20odds', '').format(league.lower(), 'ncaa%20basketball', start, end[2:])
        elif league == 'ncaafootball':
            target = SOURCE.replace('%20odds', '').format(league.lower(), 'ncaa%20football', start, end[2:])
        elif league == 'nhl':
            if year == 2021:
                target = target.split('-')[0] + '.xlsx'
        print(target)
        data = pd.read_excel(target)
        if league in ['mlb', 'nhl']:
            data = data.rename(columns={'Close': 'ML'})
        team_1 = data.iloc[0::2]
        team_2 = data.iloc[1::2]
        team_1.columns = ['{}_visitor'.format(j) for j in team_1.columns]
        team_2.columns = ['{}_home'.format(j) for j in team_2.columns]
        team = pd.concat([team_1.reset_index(drop=True), team_2.reset_index(drop=True)], axis=1)
        if league != 'mlb':
            year_cutoff = team[team['Date_visitor'].isin(range(101, 201))].index[0]
            team['year'] = pd.Series(np.where(team.index < year_cutoff, start, end))
            team['season'] = '{}-{}'.format(start, end)
        else:
            team['year'] = start
            team['season'] = '{}-{}'.format(start, start)
        if (league == 'nhl') & (int(start) > pd.datetime.now().year):
            import ipdb;
            ipdb.set_trace()
        team['date'] = pd.to_datetime(team['year'] + team['Date_visitor'].astype(str).str.zfill(4))
        return team[COLS]

    @classmethod
    def rebuild(cls, league: str = 'nba'):
        records = []
        for year in sorted(range(START, END + 1)):
            try:
                records.append(Data.format_games(year=year, league=league))
            except Exception as e:
                print('Could not get data for season {}-{} due to {}'.format(year, year + 1, e))
                pass
        data = pd.concat(records, axis=0).reset_index(drop=True)
        for j in [HOME, VISITOR]:
            data['ML_{}'.format(j)] = data['ML_{}'.format(j)].astype(str).str.replace('pk', 'NL')
            data = data[data['ML_{}'.format(j)] != 'NL']
            data = data[data['ML_{}'.format(j)] != '-']
            data['ML_{}'.format(j)] = data['ML_{}'.format(j)].astype(float)
            data = data[data['ML_{}'.format(j)].abs() > 0]
            data['price_{}'.format(j)] = data['ML_{}'.format(j)].apply(Odds.american_to_decimal)
            data['odds_{}'.format(j)] = data['price_{}'.format(j)].apply(Odds.decimal_to_probability)
        data['underdog'] = pd.Series(np.where(data.ML_home > data.ML_visitor, HOME, VISITOR), index=data.index)
        data['Final_home'] = data['Final_home'].astype(str).str.replace('F', '')
        data = data[data['Final_home'] != '']
        data['Final_home'] = data['Final_home'].astype(float)
        data['Final_visitor'] = data['Final_visitor'].astype(str).str.replace('F', '')
        data = data[data['Final_visitor'] != '']
        data['Final_visitor'] = data['Final_visitor'].astype(float)
        data['winner'] = pd.Series(np.where(data.Final_visitor > data.Final_home, VISITOR, HOME), index=data.index)
        data['win_underdog'] = pd.Series(np.where(data.underdog == data.winner, 1, -1), index=data.index)
        data['win_favorite'] = pd.Series(np.where(data.underdog == data.winner, -1, 1), index=data.index)
        data['price_underdog'] = pd.Series(np.where(data.underdog == VISITOR, data.price_visitor, data.price_home),
                                           index=data.index)
        data['price_favorite'] = pd.Series(np.where(data.underdog == HOME, data.price_visitor, data.price_home), index=data.index)
        data['ML_underdog'] = pd.Series(np.where(data.underdog == VISITOR, data.ML_visitor, data.ML_home), index=data.index)
        data['ML_favorite'] = pd.Series(np.where(data.underdog == HOME, data.ML_visitor, data.ML_home), index=data.index)
        data['underdog_profit'] = pd.Series(np.where(data.win_underdog == 1, data.price_underdog, 0), index=data.index)-1
        data['favorite_profit'] = pd.Series(np.where(data.win_favorite == 1, data.price_favorite, 0), index=data.index)-1
        data[data['price_underdog'] > data['price_favorite']]
        return data
