from utils import *

__author__ = 'kq'


def normalize_game_count(data: pd.DataFrame) -> Dict[str, float]:
    """
    Normalization of game count to account for variance in sample size
    :param data:
    :return:
    """
    counter = data.groupby(SPORT).date.count()
    return 1 / (counter / counter.max())


def long_underdog(data: pd.DataFrame) -> None:
    """
    Constant investment long-underdog strategy
    Exclude NCAAB for figure purposes
    :param data:
    :return:
    """
    data = data[data.sport != 'NCAAB']
    profit = data.set_index([DATE, SPORT]).underdog_profit.sum(level=[0, 1])
    counter = normalize_game_count(data=data)
    profit = profit.unstack().mul(counter).cumsum().fillna(method='ffill')
    profit.index = profit.index.astype(str)
    highlight_end, highlight_start = len(profit), len(profit) - len(profit[profit.index >= COVID])
    profit.plot(figsize=FIGURE_SIZE)
    plt.xticks(rotation=ROTATION, size=LABEL_SIZE)
    plt.xlim(0, round(len(profit), -3))
    plt.axvspan(highlight_start, highlight_end, color='red', alpha=0.1)
    plt.legend()
    plt.title('Long Underdog by Pro Sport')
    plt.ylabel('Cumulative Return [$]')
    plt.xlabel(DATE.title())
    plt.savefig(FIGURES.format('long_underdog'))
    plt.show()
    
def win_probability(ml_odds):
    if ml_odds<=-100:
        p = -ml_odds/(100-ml_odds)
    elif ml_odds>=100:
        p = 100/(100+ml_odds)
    else:
        p = float("nan")
    return p

def underdog_profit(ml_odds):
    if ml_odds <=-100:
        profit = 100/(-ml_odds)
    elif ml_odds>=100:
        profit = ml_odds/100
    else:
        profit = float("nan")
    return profit

#Holm-Bonferronni test
def holm_bonferonni(sports,pvals,alpha=0.05):
    df_pval = pd.DataFrame({'sport':sports,'pval':pvals})
    df_pval.sort_values(by = ['pval'],ascending = True, inplace = True)
    df_pval
    m = len(df_pval)
    k = 0
    rejects = []
    thresholds = []
    for index, row in df_pval.iterrows():
        k+=1
        threshold = alpha/(m+1-k)
        if row.pval <= threshold:
            reject = 1
        else:
            reject = 0
        print(f"{k}: {row.sport}: pval = {row.pval:.4f}, threshold = {threshold:.4f}, reject ={reject}")
        rejects.append(reject)
        thresholds.append(threshold)
    df_pval['reject'] =  rejects
    df_pval['holm_bonf_threshold'] =  threshold
    return df_pval

def prob_bin(p,bin_width=0.1):
    pmax = 0.5
    plist = list(np.arange(0,pmax+bin_width,bin_width))
    for pl in plist:
        if p<=pl:
            y = np.round(pl*100)/100
            break
        else:
            y = 1
    return y

def highlight_max(x: List[Any]) -> List[str]:
    return ['font-weight: bold' if j == np.max(x) else '' for j in x]

def sharpe(x: pd.Series) -> float:
    return x.median() / x.mad()

def bernoulli_weight(p_u: float) -> float:
    return np.sqrt(p_u * (1-p_u))

def inverse_bernoulli_weight(p_u: float) -> float:
    return 1 / np.sqrt(p_u * (1-p_u))

def moneyline_weight(p_u: float) -> float:
    return np.sqrt((1 - p_u) / p_u)

def inverse_moneyline_weight(p_u: float) -> float:
    return np.sqrt(p_u / (1 - p_u))

def probability_weight(p_u: float) -> float:
    return p_u

def inverse_probability_weight(p_u: float) -> float:
    return 1 / p_u

def uniform_weight(p_u: float) -> float:
    return 1

def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]




def performance(data: pd.DataFrame, 
                weighting: str,
                initial: float = 100, 
                risky_weight: float = 0.7) -> pd.Series:
    
    # Make sure the specified weighting is available
    assert(weighting in SCHEMES)
    lambda_r, lambda_f = np.round(risky_weight, 1), np.round(1 - risky_weight, 1)
    print(r'Testing {} scheme with $\lambda$ = {}'.format(weighting, lambda_r))
    
    
    # Assign initial parameters
    data['initial'] = initial
    data['risk'] = risky_weight
    data['weight'] = data.prob_underdog.apply(eval('{}_weight'.format(weighting)))
    
    # m_t
    weighted_profit = data.groupby('date').apply(
        lambda x: (x.weight / x.weight.sum()) * (x.underdog_profit))
    unit_game_profit = weighted_profit.reset_index(level=0, drop=True)
    unit_daily_profit = weighted_profit.sum(level=0)
    unit_daily_profit.name = 'unit_daily_profit'
    unit_game_profit.name = 'unit_game_profit'
    
    # update bankroll
    df = pd.concat([pd.merge(data[['date', 'initial', 'risk', 'weight']], unit_daily_profit, on ='date'), unit_game_profit], axis=1)
    daily = df.groupby('date').first().reset_index()
    daily['bankroll'] = ((daily.initial * daily.risk) * (daily.unit_daily_profit.shift())).fillna(initial).cumsum()
    daily = daily.set_index('date').bankroll
    drawdown = daily[daily <= 0]
    
    # shutdown if drawdown
    if len(drawdown) > 0:
        daily.loc[drawdown.index[0]:] = 0
    return daily

