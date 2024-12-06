import pandas as pd
import matplotlib.pyplot as plt
"""
sub-event types 'full time', 'goal', 'half time', 'kick off', 'other', 'owngoal', 'penalty', 'red card', 'yellow card'
"""
# Charger les données
df = pd.read_csv("data/embedded_data/very_simple_embedding/train_tweets.csv")
df = df[df['MatchID'] == 14]

# Configurer la taille de la figure
plt.style.use('bmh')
plt.rc('axes', facecolor='none')
plt.rc('figure', figsize=(20, 6))  # Agrandir la figure


M = df["count_capital_letter"].mean()

# Tracer les courbes des métriques
plt.plot(df["PeriodID"], df["count_capital_letter"]-M, label='count_capital_letter_moins_mean')


# Filtrer les points où EventType == 1
event_indices = df[df["EventType"] == 1]["PeriodID"]

# Ajouter les points rouges
plt.scatter(event_indices, [0] * len(event_indices), color='red', label='EventType=1')

# Ajouter des graduations plus fines à l'axe des abscisses
plt.xticks(range(min(df["PeriodID"]), max(df["PeriodID"]) + 1, 5))

# Ajouter une légende
plt.legend()

# Sauvegarder l'image
plt.savefig("plot_count_capital_letter", dpi=300)
plt.close()


plt.style.use('bmh')
plt.rc('axes', facecolor='none')
plt.rc('figure', figsize=(20, 6))  # Agrandir la figure

M = df["count_exclamation"].mean()

# Tracer les courbes des métriques

plt.plot(df["PeriodID"], df["count_exclamation"]-M, label='count_exclamation_less_mean')


# Filtrer les points où EventType == 1
event_indices = df[df["EventType"] == 1]["PeriodID"]

# Ajouter les points rouges
plt.scatter(event_indices, [0] * len(event_indices), color='red', label='EventType=1')

# Ajouter des graduations plus fines à l'axe des abscisses
plt.xticks(range(min(df["PeriodID"]), max(df["PeriodID"]) + 1, 5))

# Ajouter une légende
plt.legend()

# Sauvegarder l'image
plt.savefig("plot_count_exclamation", dpi=300)
plt.close()


plt.style.use('bmh')
plt.rc('axes', facecolor='none')
plt.rc('figure', figsize=(20, 6))  # Agrandir la figure

M = df["count_tweet_lenght"].mean()

# Tracer les courbes des métriques

plt.plot(df["PeriodID"], df["count_tweet_lenght"]-M, label='count_tweet_lenght_less_mean')


# Filtrer les points où EventType == 1
event_indices = df[df["EventType"] == 1]["PeriodID"]

# Ajouter les points rouges
plt.scatter(event_indices, [0] * len(event_indices), color='red', label='EventType=1')

# Ajouter des graduations plus fines à l'axe des abscisses
plt.xticks(range(min(df["PeriodID"]), max(df["PeriodID"]) + 1, 5))

# Ajouter une légende
plt.legend()

# Sauvegarder l'image
plt.savefig("plot_count_tweet_lenght", dpi=300)
plt.close()


plt.style.use('bmh')
plt.rc('axes', facecolor='none')
plt.rc('figure', figsize=(20, 6))  # Agrandir la figure

M = df["count_tweet_per_period"].mean()

# Tracer les courbes des métriques

plt.plot(df["PeriodID"], df["count_tweet_per_period"]-M, label='count_tweet_per_period_less_mean')

# Filtrer les points où EventType == 1
event_indices = df[df["EventType"] == 1]["PeriodID"]

# Ajouter les points rouges
plt.scatter(event_indices, [0] * len(event_indices), color='red', label='EventType=1')

# Ajouter des graduations plus fines à l'axe des abscisses
plt.xticks(range(min(df["PeriodID"]), max(df["PeriodID"]) + 1, 5))

# Ajouter une légende
plt.legend()

# Sauvegarder l'image
plt.savefig("plot_count_tweet_per_period", dpi=300)
plt.close()
