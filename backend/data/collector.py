# file to execute from root with `python -m backend.data.collector` 

import requests
import json
import os
from PIL import Image
import time
from pathlib import Path
import pandas as pd

def extract_clean_sprites(sprites):
    """
    Récupère les sprites pertinents :
    - /other/official-artwork/front_default
    - /sprites/front_default
    - /sprites/versions/**/front_default (sauf gen1 et gen2)
    """
    selected_images = []

    # 1. OFFICIAL ARTWORK
    official_url = sprites.get("other", {}).get("official-artwork", {}).get("front_default")
    if official_url:
        selected_images.append((official_url, "official"))

    # 2. ROOT front_default (souvent une version standard)
    root_front = sprites.get("front_default")
    if root_front:
        selected_images.append((root_front, "default"))

    # 3. VERSIONS (on évite gen I et II car trop pixelisées)
    versions = sprites.get("versions", {})
    for generation, games in versions.items():
        if generation in {"generation-i", "generation-ii"}:
            continue  # ❌ exclude low-res gens

        for game_name, game_sprites in games.items():
            if game_name in {"emerald", "platinum", "black-white", "x-y", "icons"}:
                continue # exclude duplicate for instance emerald sprite is the same sprite as ruby-sapphire

            front_default_url = game_sprites.get("front_default")
            if front_default_url:
                label = f"{generation}_{game_name}"
                selected_images.append((front_default_url, label))

    return selected_images



class PokemonDataCollector:
    def __init__(self, output_dir="backend/data/pokemon_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.metadata = []
        
    def get_pokemon_list(self, limit=151):
        """Get pokemon list from PokeAPI"""
        print(f"Get list of {limit} first Pokemons...")
        url = f"https://pokeapi.co/api/v2/pokemon?limit={limit}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['results']
        return []
    
    def get_pokemon_details(self, pokemon_url):
        """Get details of a specific Pokémon"""
        try:
            response = requests.get(pokemon_url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error while fetching details: {e}")
            return None
    
    def get_pokemon_species(self, species_url):
        """Get species info (gen, color, etc.)"""
        try:
            response = requests.get(species_url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération de l'espèce: {e}")
            return None
    
    def download_image(self, image_url, pokemon_name, image_type="official"):
        """Download and process Pokemon images"""
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                # file name
                filename = f"{pokemon_name}_{image_type}.png"
                filepath = self.images_dir / filename

                # save original file
                with open(filepath, 'wb') as f:
                    f.write(response.content)

                try:
                    img = Image.open(filepath)

                    # If image has alpha channel, add black background
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        alpha = img.convert('RGBA')
                        background = Image.new('RGB', alpha.size, (0, 0, 0))  # black background
                        img = Image.alpha_composite(background.convert('RGBA'), alpha).convert('RGB')
                    else:
                        img = img.convert('RGB')

                    # Resize and save as JPEG
                    img = img.resize((128, 128), Image.Resampling.LANCZOS)
                    img.save(filepath.with_suffix('.jpg'), 'JPEG', quality=95)

                    # Delete original PNG
                    filepath.unlink()

                    return str(filepath.with_suffix('.jpg'))

                except Exception as e:
                    print(f"Error while processing {filename}: {e}")
                    return None

            return None

        except Exception as e:
            print(f"Error while downloading image: {e}")
            return None


    
    def extract_metadata(self, pokemon_data, species_data):
        """Extract metadata from Pokemon"""
        metadata = {
            'name': pokemon_data['name'],
            'id': pokemon_data['id'],
            'height': pokemon_data['height'],
            'weight': pokemon_data['weight'],
            'types': [t['type']['name'] for t in pokemon_data['types']],
            'primary_type': pokemon_data['types'][0]['type']['name'],
            'stats': {stat['stat']['name']: stat['base_stat'] 
                     for stat in pokemon_data['stats']},
        }
        
        if species_data:
            metadata.update({
                'generation': species_data['generation']['name'],
                'generation_id': species_data['generation']['url'].split('/')[-2],
                'color': species_data['color']['name'],
                'habitat': species_data['habitat']['name'] if species_data['habitat'] else None,
                'is_legendary': species_data['is_legendary'],
                'is_mythical': species_data['is_mythical'],
            })
        
        return metadata
    
    def collect_pokemon_data(self, limit=151):
        """Main fonction to collect Pokemon data"""
        pokemon_list = self.get_pokemon_list(limit)
        
        for i, pokemon in enumerate(pokemon_list):
            print(f"Processing {i+1}/{len(pokemon_list)}: {pokemon['name']}")
            
            pokemon_data = self.get_pokemon_details(pokemon['url'])
            if not pokemon_data:
                continue

            species_data = self.get_pokemon_species(pokemon_data['species']['url'])
            metadata = self.extract_metadata(pokemon_data, species_data)
            sprites = pokemon_data['sprites']
            image_paths = []
            
            relevant_images = extract_clean_sprites(sprites)

            image_entries = []
            for image_url, image_type in relevant_images:
                img_path = self.download_image(image_url, pokemon['name'], image_type)
                if img_path:
                    is_sprite = image_type != "official"
                    image_entries.append({
                        "image_path": img_path,
                        "image_type": image_type,
                        "is_sprite": is_sprite
                    })

            metadata["images"] = image_entries
            metadata["num_images"] = len(image_entries)
            self.metadata.append(metadata)

            # time.sleep(0.1)
        
        self.save_metadata()
        print(f"Collect finished ! {len(self.metadata)} Pokemon collected.")
    
    def save_metadata(self):
        """Save data in JSON and CSV"""
        json_path = self.output_dir / "metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Flatten to CSV
        df_data = []
        for pokemon in self.metadata:
            for image in pokemon.get("images", []):
                df_data.append({
                    'name': pokemon['name'],
                    'id': pokemon['id'],
                    'primary_type': pokemon['primary_type'],
                    'types': ','.join(pokemon['types']),
                    'color': pokemon.get('color', 'unknown'),
                    'generation': pokemon.get('generation', 'unknown'),
                    'generation_id': pokemon.get('generation_id', 0),
                    'is_legendary': pokemon.get('is_legendary', False),
                    'is_mythical': pokemon.get('is_mythical', False),
                    'height': pokemon['height'],
                    'weight': pokemon['weight'],
                    'image_path': image["image_path"],
                    'image_type': image["image_type"],
                    'is_sprite': image["is_sprite"]
                })

        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"Metadata saved in {json_path} and {csv_path}")

    
    def create_simple_descriptions(self):
        """Generate simple descriptions for each Pokemon (usefull for training conditionnal models)"""
        descriptions = []
        
        for pokemon in self.metadata:
            base_desc = f"a {pokemon['color']} {pokemon['primary_type']} pokemon"
            
            # Other attributes
            attrs = []
            if pokemon.get('is_legendary'):
                attrs.append("legendary")
            if pokemon.get('is_mythical'):
                attrs.append("mythical")
            if len(pokemon['types']) > 1:
                attrs.append(f"{pokemon['types'][1]} type")
            
            if attrs:
                base_desc += f" that is {', '.join(attrs)}"
            
            descriptions.append({
                'name': pokemon['name'],
                'simple_description': base_desc,
                'detailed_description': f"a {pokemon['color']} {'/'.join(pokemon['types'])} pokemon from generation {pokemon.get('generation_id', 1)}"
            })
        
        # Sauvegarde
        desc_path = self.output_dir / "descriptions.json"
        with open(desc_path, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)
        
        return descriptions


if __name__ == "__main__":
    collector = PokemonDataCollector()
    
    # Collecte pokemon
    collector.collect_pokemon_data(limit=1)
    
    # Generate simple descriptions
    collector.create_simple_descriptions()
    
    print("Dataset created successfully !")
    print(f"Structure:")
    print(f"- Images: {collector.images_dir}")
    print(f"- Metadata: {collector.output_dir}/metadata.json")
    print(f"- CSV Dataset: {collector.output_dir}/dataset.csv")
    print(f"- JSON Descriptions: {collector.output_dir}/descriptions.json")