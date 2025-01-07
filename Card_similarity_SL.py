import pandas as pd
import numpy as np
import requests
import streamlit as st
import ast
import gzip
from io import BytesIO
import gdown


def get_card_details(card_name):
    """
    Fetches the image URL, price of the cheapest version of a card, and its Scryfall URI
    from the Scryfall API. Skips printings where prices are unavailable.

    Args:
        card_name (str): The name of the card to search for.

    Returns:
        tuple: A tuple containing the image URL (str), price (float), and Scryfall URI (str).
    """
    api_url = f"https://api.scryfall.com/cards/search?q=!\"{card_name}\"&unique=prints&order=usd&dir=asc"
    response = requests.get(api_url)

    if response.status_code == 200:
        card_data = response.json()
        if 'data' in card_data and card_data['data']:
            for printing in card_data['data']:
                # Check if price is available
                price = printing['prices']['usd'] if 'prices' in printing and printing['prices']['usd'] else None
                if price:  # If price is available, return this printing
                    image_url = printing['image_uris']['normal'] if 'image_uris' in printing else None
                    scryfall_uri = printing['scryfall_uri'] if 'scryfall_uri' in printing else None
                    return image_url, float(price), scryfall_uri

    # If no valid price is found, return None values
    return None, None, None



def find_similar_cards_with_filters(df, card_name, n_neighbors=5, n_dimensions=10, 
                                    metric="cosine", deck_colors=None, type_line=None, commander_legal=False):
    """
    Find the most similar cards to a given card using SVD dimensions with filters.

    Args:
        df (pd.DataFrame): The DataFrame containing card data and SVD dimensions.
        card_name (str): The name of the card to search for.
        n_neighbors (int): The number of similar cards to return.
        n_dimensions (int): The number of SVD dimensions to consider.
        metric (str): Distance metric to use ('cosine' or 'euclidean').
        deck_colors (str, optional): Specify the color identity of the deck (e.g., 'WB' for white and black).
        type_line (str, optional): Filter by type line (e.g., 'Creature').
        commander_legal (bool): Whether to filter for cards legal in Commander format.

    Returns:
        pd.DataFrame: DataFrame with the top n_neighbors similar cards and their details.
    """
    # Filter SVD columns dynamically
    svd_columns = [f"SVD_{i}" for i in range(1, n_dimensions + 1)]
    if not all(col in df.columns for col in svd_columns):
        raise ValueError(f"Not all SVD dimensions up to {n_dimensions} are present in the DataFrame")

    # Ensure the card exists in the DataFrame (case-insensitive)
    card_name = card_name.lower()
    df['name'] = df['name'].str.lower()

    if card_name not in df['name'].values:
        raise ValueError(f"Card '{card_name}' not found in the DataFrame")

    # Apply filters to the DataFrame
    filtered_df = df.copy()

    # Filter by deck colors (subset rule), but always include colorless cards
    if deck_colors:
        deck_colors_set = set(deck_colors.upper())  # Convert deck colors to uppercase for consistency

        # Parse the color_identity column from string to a list
        filtered_df['color_identity'] = filtered_df['color_identity'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )

        # Apply the filter
        def is_valid_color_identity(x):
            if not x or x == ["C"]:  # Include colorless cards
                return True
            if isinstance(x, list):
                return set(x).issubset(deck_colors_set)  # Subset match
            return False  # Exclude invalid entries

        filtered_df = filtered_df[filtered_df['color_identity'].apply(is_valid_color_identity)]

    # Filter by type line
    if type_line:
        type_line = type_line.lower()
        filtered_df['type_line'] = filtered_df['type_line'].str.lower()
        filtered_df = filtered_df[filtered_df['type_line'].str.contains(type_line, case=False)]

    # Filter for Commander legality
    if commander_legal:
        if 'legal' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['legal'].apply(lambda x: 'commander' in ast.literal_eval(x))]
        else:
            raise ValueError("The dataset does not contain legality information for filtering.")

    # Check if filtered DataFrame is empty
    if filtered_df.empty:
        raise ValueError("No cards match the specified filters")

    # Get the target card's SVD vector
    target_card = df[df['name'] == card_name]
    target_vector = target_card[svd_columns].values[0]

    # Get filtered cards' SVD vectors
    filtered_vectors = filtered_df[svd_columns].values

    # Compute distances
    if metric == "cosine":
        target_norm = np.linalg.norm(target_vector)
        filtered_norms = np.linalg.norm(filtered_vectors, axis=1)
        cosine_similarities = np.dot(filtered_vectors, target_vector) / (filtered_norms * target_norm)
        distances = 1 - cosine_similarities
    elif metric == "euclidean":
        distances = np.linalg.norm(filtered_vectors - target_vector, axis=1)
    else:
        raise ValueError("Metric must be 'cosine' or 'euclidean'")

    # Add distances and similarities to the filtered DataFrame
    filtered_df = filtered_df.copy()
    filtered_df['distance'] = distances

    # Add percentage similarity for better interpretation
    filtered_df['similarity (%)'] = (1 - distances) * 100

    # Optional: Apply a nonlinear transformation to emphasize differences
    filtered_df['distance (transformed)'] = distances ** 2

    # Sort by distance and exclude the target card itself
    similar_cards = filtered_df[filtered_df['name'] != card_name].sort_values('distance').head(n_neighbors)

    return similar_cards[['name', 'oracle_text', 'color_identity', 'type_line', 'distance', 'similarity (%)', 'distance (transformed)']]



#https://drive.google.com/file/d/1R47yAGvLk1EsMJ80Sv2NbqNV6rO1qNw2/view?usp=drive_link

def main():
    st.title("Magic: The Gathering Card Similarity Finder")
    
    #@st.cache_data
    def load_compressed_data_with_gdown(file_id):
        # Google Drive download URL for gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download the file using gdown
        output = BytesIO()
        gdown.download(url, output, quiet=False)
        output.seek(0)  # Move to the start of the file
        
        # Load the compressed file
        with gzip.open(output, "rt") as f:
            return pd.read_csv(f)

    # Replace with your file ID
    file_id = "1R47yAGvLk1EsMJ80Sv2NbqNV6rO1qNw2"
    mtg = load_compressed_data_with_gdown(file_id)

    mtg.head()


    # Sidebar inputs
    st.sidebar.header("Filter Options")
    card_name = st.sidebar.text_input("Enter the name of a card:", "Counterspell", autocomplete="on")
    n_neighbors = st.sidebar.slider("Number of similar cards to display:", 1, 20, 5)
    deck_colors = st.sidebar.text_input("Deck colors (e.g., 'WB' for white-black):", "")
    type_line = st.sidebar.text_input("Type line filter (e.g., 'Instant', 'Creature'):", "")
    commander_legal = st.sidebar.checkbox("Show only cards legal in Commander", value=False)

    # Display the input card's image and name in the sidebar
    input_image_url, _, input_scryfall_uri = get_card_details(card_name)
    if input_image_url and input_scryfall_uri:
        st.sidebar.markdown(
            f'<a href="{input_scryfall_uri}" target="_blank"><img src="{input_image_url}" width="250"></a>',
            unsafe_allow_html=True,
        )
        st.sidebar.write(f"**Input Card:** {card_name}")

    if st.sidebar.button("Find Similar Cards"):
        try:
            # Fetch the details of the selected card
            input_image_url, input_price, input_scryfall_uri = get_card_details(card_name)
    
         
            if input_price:
                st.sidebar.write(f"**Cheapest Printing Price:** ${input_price:.2f}")
            else:
                st.sidebar.write("**Cheapest Printing Price:** Not available")

            # Find similar cards
            similar_cards = find_similar_cards_with_filters(
                df=mtg,
                card_name=card_name,
                n_neighbors=n_neighbors,
                metric="cosine",
                deck_colors=deck_colors,
                type_line=type_line,
                commander_legal=commander_legal
            )
    
            st.write("### Similar Cards")
            for _, row in similar_cards.iterrows():
                image_url, price, scryfall_uri = get_card_details(row["name"])
                col1, col2 = st.columns([1, 2])
                with col1:
                    if image_url and scryfall_uri:
                        st.markdown(
                            f'<a href="{scryfall_uri}" target="_blank"><img src="{image_url}" width="250"></a>',
                            unsafe_allow_html=True,
                        )
                with col2:
                    st.write(f"**Name:** {row['name']}")
                    st.write(f"**Oracle Text:** {row['oracle_text']}")
                    st.write(f"**Color Identity:** {row['color_identity']}")
                    st.write(f"**Type Line:** {row['type_line']}")
                    st.write(f"**Similarity:** {row['similarity (%)']:.2f}%")
                    st.write(f"**Distance (Transformed):** {row['distance (transformed)']:.4f}")
                    if price:
                        st.write(f"**Price:** ${price}")
                        if input_price:
                            price_difference = float(price) - input_price
                            # Style the price difference text
                            if price_difference < 0:
                                st.markdown(
                                    f"<span style='color: green;'>**Price Difference:** ${price_difference:.2f}</span>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<span style='color: red;'>**Price Difference:** +${price_difference:.2f}</span>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.write("**Price Difference:** Not available")
                    else:
                        st.write("**Price:** Not available")
    
        except ValueError as e:
            st.error(str(e))






if __name__ == "__main__":
    main()

