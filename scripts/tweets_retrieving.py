import os
import pandas as pd
import re
import unicodedata
from html import unescape
import json


def extract_members_from_data(path):
    """
    Extract Twitter account data for congressional members from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing member information and Twitter accounts.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: screen_name, account_type, party, chamber, and name
        for each valid congressional member account.
    """
    with open(path, "r") as f:
        data = json.load(f)

    records = []
    for entry in data:
        if entry.get("type") != "member":
            continue  # Skip non-member entries

        for acc in entry.get("accounts", []):
            if acc.get("account_type") != "office":
                continue  # Skip non-office accounts
            records.append({
                "screen_name": acc.get("screen_name", "").lower(),
                "account_type": acc.get("account_type", "MISSING"),
                "party": entry.get("party", "MISSING"),
                "chamber": entry.get("chamber", "MISSING"),
                "name": entry.get("name", "MISSING")
            })

    members_df = pd.DataFrame(records)
    print("Members extracted:", len(members_df))
    return members_df


def clean_text(text):
    """
    Clean and normalize tweet text by removing noise and unwanted characters.

    Parameters
    ----------
    text : str
        The raw tweet text to be cleaned.

    Returns
    -------
    str
        The cleaned and normalized tweet text.
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = unescape(text)

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\(\d+/[x\d]+\)", "", text)  # Remove thread markers
    text = re.sub(r"\(Fin\)", "", text, flags=re.I)  # Remove endings
    text = re.sub(r"^RT @\w+: ", "", text)  # Remove retweet headers
    text = re.sub(r"QT @\w+", "", text)  # Remove quoted tweet intros
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"http\S+", "", text)  # Remove URLs

    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\u00a0\u00ad\uFEFF]", "", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")

    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def process_tweets_range(
    data_dir,
    members_df,
    date_range=("2020-12-01", "2021-01-05"),
    hour_range=None,
    output_dir="./df_clean/",
):
    """
    Filter and save congressional tweets within a specified date and time range.

    Parameters
    ----------
    data_dir : str
        Directory containing JSON files with tweet data.
    members_df : pd.DataFrame
        DataFrame containing metadata of valid congressional members.
    date_range : tuple of str
        Start and end dates in the format ('YYYY-MM-DD', 'YYYY-MM-DD').
    hour_range : tuple of int or None
        Optional range of hours (start_hour, end_hour) to filter tweets.
    output_dir : str
        Directory to save the resulting CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame of cleaned and filtered tweets within the specified range.
    """
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    start_date = pd.to_datetime(date_range[0], utc=True)
    end_date = pd.to_datetime(date_range[1], utc=True)

    valid_screen_names = set(members_df["screen_name"].str.lower())
    collected_tweets = []

    for file in sorted(os.listdir(data_dir)):
        if not file.endswith(".json") or "historical-users" in file:
            continue

        try:
            file_date = pd.to_datetime(file.replace(".json", ""), utc=True)
        except:
            continue
        if not (start_date <= file_date <= end_date):
            continue

        file_path = os.path.join(data_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tweets = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping bad file: {file}")
            continue

        for tweet in tweets:
            screen = tweet.get("screen_name", "").lower()
            if screen not in valid_screen_names:
                continue

            tweet["text"] = clean_text(tweet.get("text", ""))

            try:
                tweet_time = pd.to_datetime(tweet.get("time"), utc=True)
            except:
                tweet_time = file_date

            if not (start_date <= tweet_time <= end_date):
                continue

            if hour_range:
                hour = tweet_time.hour
                start_hr, end_hr = hour_range
                if start_hr < end_hr:
                    if not (start_hr <= hour < end_hr):
                        continue
                else:
                    if not (hour >= start_hr or hour < end_hr):
                        continue

            tweet["time"] = tweet_time
            collected_tweets.append(tweet)

    print(f"Collected {len(collected_tweets)} tweets from {date_range[0]} to {date_range[1]}")

    df = pd.DataFrame(collected_tweets)
    df["screen_name"] = df["screen_name"].str.lower()

    df = df.merge(members_df, on="screen_name", how="left")
    df = df[df["party"].isin(["D", "R"])].copy()

    output_file = f"tweets_{date_range[0]}_{date_range[1]}"
    if hour_range:
        output_file += f"_{hour_range[0]}to{hour_range[1]}"
    df.to_csv(os.path.join(output_dir, output_file + ".csv"), index=False)

    print(f"Saved to {output_file}.csv")
    return df

