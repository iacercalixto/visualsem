# Creating VisualSem from scratch
This code is ported from [Houda Alberts](https://github.com/houda96/VisualSem/tree/master/dataset_creation) master thesis work.

## Disclaimer
Creating VisualSem is a time-consuming process; it can take up to a few days to download and process all the nodes, edges, and images. Moreover, image files take up TBs of memory. Please make sure you have the needed space available.

## Usage

### Before you start
Create an Python 3.6 environment and install `requirements.txt`.

    python3.6 -m venv venv_name
    pip install -r requirements.txt
    source venv_name

Before you start, also make sure:

- You create the data subdirectory: `mkdir ./data`
- Create subdirectory where to download images: `mkdir ./data/visualsem_images`.
- Download [nodes_1000.json](https://surfdrive.surf.nl/files/index.php/s/8VrHn8TDPwqMiat) and store it in `./data`.

### Prerequisites
To generate VisualSem from scratch, you must first create an account in the [BabelNet website](https://babelnet.org/) and download a local BabelNet index (we specifically use BabelNet v4.0). Please follow the instructions in the BabelNet website to how to download the index (if in doubt, you can [start here](https://babelnet.org/guide#HowcanIdownloadtheBabelNetindices?)).

Please then follow [our guide](https://github.com/robindv/babelnet-api) on how to set up the local BabelNet index to be used to create VisualSem. By following the guide you will serve the index on port 8080 in the same local machine where you will run scripts below.

Besides having BabelNet configured, our pipeline also uses code ported from the [imagi-filter repository](https://github.com/houda96/imagi-filter) for filtering out noisy images. Run the following to download the weights of a ResNet-152 pretrained to filter noisy images.

- Download pretrained ResNet152 model [here](https://surfdrive.surf.nl/files/index.php/s/ipyfk9iJcWvZYYk).
- Move the model checkpoint inside the `./data/` directory.

### Generating VisualSem

You can call any script below with the `--help` flag to check what parameters can be changed. However, if you change any parameters you will need to manually find generated intermediate files and set the right path in further scripts when needed, since some files generated are used later in other scripts.

1. We start by extracting nodes. Please beware of possibly long runtimes.

```python
python extract_nodes.py
```

Optionally, you can set many flags when running `extract_nodes.py`.

```python
python extract_nodes.py
    --initial_node_file data/nodes_1000.json
    --relations_file data/rels.json # where to create rels.json
    --store_steps_nodes path/to/storage # where to store temporary/debugging files
    --k neighboring_nodes_max # max number of neighbour nodes per relation considered at each step
    --min_ims number_of_images # minimum number of images required per node
    --M num_iterations # number of iterations to run
    --placement location_babelnet_api # where BabelNet is running, by default "localhost:8080"
```

2. Next, we create and save relations between nodes.

```python
python store_edg_info.py
```

3. We gather some additional images for the initial core nodes, which can be extended if more nodes must have images.

```python
python google_download.py
    --images_folder data/visualsem_images
    --initial_node_file data/nodes_1000.json
```

4. Since we are dealing with many images, we first partition the images in chunks to be able to either run things in parallel or have in between breaks.

```python
python image_urls.py
```

5. We use [aria2](https://aria2.github.io/) via the command-line to download images. These parameters have been optimized for our devices, please ensure that this is also suitable for the specs on your device.

```
aria2c -i data/urls_file -x 16 -j 48 -t 5 --disable-ipv6 --connect-timeout=5
```

6. We then hash images to remove duplicates.

```python
python hash_exist_images.py
```

7. We fix invalid image types to correct ones.

```
bash convert.sh data/visualsem_images/*
```

8. We resize images to reduce disk space.

```python
python resizing.py --images_folder data/visualsem_images/*
```

9. A first filtering is done by checking whether image files are correctly formatted; i.e. this step filters out ill-formatted image files.

```python
python list_good_bad_files.py --images_folder data/visualsem_images
```

10. This step classifies images as being useful or not and ouputs JSON files for later filtering. The parser can take many arguments here. *NOTE: This part follows [imagi-filter](https://github.com/houda96/imagi-filter) as explained above*

```python
python forward_pass.py --img_store data/visualsem_images
```

11. Now we filter out images that are not useful; we do not remove them, but simply do not keep them in the information system itself so that non-informative images can be inspected if necessary.

```python
python filter_images.py
```

You can optionally run this file setting the paths of the different files, if you have called scripts with non-default parameters before:

```python
python filter_images.py
    --hashes_storage /path/to/input/hash
    --dict_hashes /path/to/input/dict_hashes
    --nodes_180k /path/to/input/nodes
    --edges_180k /path/to/input/edged
    --marking_dict /path/to/input/marking_dict
    --nodes_file /path/to/output/nodes
    --edges_file /path/to/output/edges
```

After running these scripts, you will have generated the set of nodes in `./data/nodes.json`, the set of edges/tuples in `./data/tuples.json`, and the set of images associated to nodes in `./data/visualsem_images`.
