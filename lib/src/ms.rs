use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;
use rstar::RTree;
use std::cmp::max;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use crate::{img_pyramid::*, unsync::*, CoordinateTransform, Dims, SamplingMethod};

const TILING_BOUNDARY_PERCENTAGE: f32 = 0.05;

#[derive(Debug)]
pub struct GeneratorParams {
    /// How many neighboring pixels each pixel is aware of during the generation
    /// (bigger number -> more global structures are captured).
    pub(crate) nearest_neighbors: u32,
    /// How many random locations will be considered during a pixel resolution
    /// apart from its immediate neighbors (if unsure, keep same as k-neighbors)
    pub(crate) random_sample_locations: u64,
    /// The distribution dispersion used for picking best candidate (controls
    /// the distribution 'tail flatness'). Values close to 0.0 will produce
    /// 'harsh' borders between generated 'chunks'. Values  closer to 1.0 will
    /// produce a smoother gradient on those borders.
    pub(crate) cauchy_dispersion: f32,
    /// The percentage of pixels to be backtracked during each p_stage.
    /// Range (0,1).
    pub(crate) p: f32,
    /// Controls the number of backtracking stages. Backtracking prevents
    /// 'garbage' generation
    pub(crate) p_stages: i32,
    /// random seed
    pub(crate) seed: u64,
    /// controls the trade-off between guide and example map
    pub(crate) alpha: f32,
    pub(crate) max_thread_count: usize,
    pub(crate) tiling_mode: bool,
}

#[derive(Debug, Default, Clone)]
struct CandidateStruct {
    coord: (SignedCoord2D, MapId), //X, Y, and map_id
    k_neighs: Vec<(SignedCoord2D, MapId)>,
    id: (PatchId, MapId),
}

impl CandidateStruct {
    fn clear(&mut self) {
        self.k_neighs.clear();
    }
}

struct GuidesStruct<'a> {
    pub example_guides: Vec<ImageBuffer<'a>>, // as many as there are examples
    pub target_guide: ImageBuffer<'a>,        //single for final color_map
}

pub(crate) struct GuidesPyramidStruct {
    pub example_guides: Vec<ImagePyramid>, // as many as there are examples
    pub target_guide: ImagePyramid,        //single for final color_map
}

impl GuidesPyramidStruct {
    fn to_guides_struct(&self, level: usize) -> GuidesStruct<'_> {
        let tar_guide = ImageBuffer::from(&self.target_guide.pyramid[level]);
        let ex_guide = self
            .example_guides
            .iter()
            .map(|a| ImageBuffer::from(&a.pyramid[level]))
            .collect();

        GuidesStruct {
            example_guides: ex_guide,
            target_guide: tar_guide,
        }
    }
}

#[inline]
fn modulo(a: i32, b: i32) -> i32 {
    let result = a % b;
    if result < 0 {
        result + b
    } else {
        result
    }
}

// for k-neighbors
#[derive(Clone, Copy, Debug, Default)]
struct SignedCoord2D {
    x: i32,
    y: i32,
}

impl SignedCoord2D {
    fn from(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn to_unsigned(self) -> Coord2D {
        Coord2D::from(self.x as u32, self.y as u32)
    }

    #[inline]
    fn wrap(self, (dimx, dimy): (i32, i32)) -> Self {
        let mut c = self;
        c.x = modulo(c.x, dimx);
        c.y = modulo(c.y, dimy);
        c
    }
}

#[derive(Clone, Copy, Debug)]
struct Coord2D {
    x: u32,
    y: u32,
}

impl Coord2D {
    fn from(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    fn to_flat(self, dims: Dims) -> CoordFlat {
        CoordFlat(dims.width * self.y + self.x)
    }

    fn to_signed(self) -> SignedCoord2D {
        SignedCoord2D {
            x: self.x as i32,
            y: self.y as i32,
        }
    }
}
#[derive(Clone, Copy, Debug)]
struct CoordFlat(u32);

impl CoordFlat {
    fn to_2d(self, dims: Dims) -> Coord2D {
        let y = self.0 / dims.width;
        let x = self.0 - y * dims.width;
        Coord2D::from(x, y)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PatchId(u32);
#[derive(Clone, Copy, Debug, Default)]
struct MapId(u32);
#[derive(Clone, Copy, Debug, Default)]
struct Score(f32);

#[derive(Clone, Debug, Default)]
struct ColorPattern(Vec<u8>);

impl ColorPattern {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}

#[derive(Clone)]
pub(crate) struct ImageBuffer<'a> {
    buffer: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> ImageBuffer<'a> {
    #[inline]
    fn is_in_bounds(&self, coord: SignedCoord2D) -> bool {
        coord.x >= 0 && coord.y >= 0 && coord.x < self.width as i32 && coord.y < self.height as i32
    }

    #[inline]
    fn get_pixel(&self, x: u32, y: u32) -> &'a image::Rgba<u8> {
        let ind = (y as usize * self.width + x as usize) * 4;
        unsafe { &*((&self.buffer[ind..ind + 4]).as_ptr() as *const image::Rgba<u8>) }
    }

    #[inline]
    fn dimensions(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }
}

impl<'a> From<&'a image::RgbaImage> for ImageBuffer<'a> {
    fn from(img: &'a image::RgbaImage) -> Self {
        let (width, height) = img.dimensions();
        Self {
            buffer: img,
            width: width as usize,
            height: height as usize,
        }
    }
}

pub struct Generator {
    pub(crate) color_map: UnsyncRgbaImage,
    coord_map: UnsyncVec<(Coord2D, MapId)>, //list of samples coordinates from example map
    id_map: UnsyncVec<(PatchId, MapId)>,    // list of all id maps of our generated image
    pub(crate) output_size: Dims,           // size of the generated image
    unresolved: Mutex<Vec<CoordFlat>>,      //for us to pick from
    resolved: RwLock<Vec<(CoordFlat, Score)>>, //a list of resolved coordinates in our canvas and their scores
    tree_grid: TreeGrid,                       // grid of R*Trees
    locked_resolved: usize,                    //used for inpainting, to not backtrack these pixels
    input_dimensions: Vec<Dims>,
}

impl Generator {
    pub(crate) fn new(size: Dims) -> Self {
        let s = (size.width as usize) * (size.height as usize);
        let unresolved: Vec<CoordFlat> = (0..(s as u32)).map(CoordFlat).collect();
        Self {
            color_map: UnsyncRgbaImage::new(image::RgbaImage::new(size.width, size.height)),
            coord_map: UnsyncVec::new(vec![(Coord2D::from(0, 0), MapId(0)); s]),
            id_map: UnsyncVec::new(vec![(PatchId(0), MapId(0)); s]),
            output_size: size,
            unresolved: Mutex::new(unresolved),
            resolved: RwLock::new(Vec::new()),
            tree_grid: TreeGrid::new(size.width, size.height, max(size.width, size.height), 0, 0),
            locked_resolved: 0,
            input_dimensions: Vec::new(),
        }
    }

    pub(crate) fn new_from_inpaint(
        size: Dims,
        inpaint_map: image::RgbaImage,
        color_map: image::RgbaImage,
        color_map_index: usize,
    ) -> Self {
        let inpaint_map =
            if inpaint_map.width() != size.width || inpaint_map.height() != size.height {
                image::imageops::resize(
                    &inpaint_map,
                    size.width,
                    size.height,
                    image::imageops::Triangle,
                )
            } else {
                inpaint_map
            };

        let color_map = if color_map.width() != size.width || color_map.height() != size.height {
            image::imageops::resize(
                &color_map,
                size.width,
                size.height,
                image::imageops::Triangle,
            )
        } else {
            color_map
        };

        let s = (size.width as usize) * (size.height as usize);
        let mut unresolved: Vec<CoordFlat> = Vec::new();
        let mut resolved: Vec<(CoordFlat, Score)> = Vec::new();
        let mut coord_map = vec![(Coord2D::from(0, 0), MapId(0)); s];
        let tree_grid = TreeGrid::new(size.width, size.height, max(size.width, size.height), 0, 0);
        //populate resolved, unresolved and coord map
        for (i, pixel) in inpaint_map.pixels().enumerate() {
            if pixel[0] < 255 {
                unresolved.push(CoordFlat(i as u32));
            } else {
                resolved.push((CoordFlat(i as u32), Score(0.0)));
                let coord = CoordFlat(i as u32).to_2d(size);
                coord_map[i] = (coord, MapId(color_map_index as u32)); //this absolutely requires the input image and output image to be the same size!!!!
            }
        }

        let locked_resolved = resolved.len();
        Self {
            color_map: UnsyncRgbaImage::new(color_map),
            coord_map: UnsyncVec::new(coord_map),
            id_map: UnsyncVec::new(vec![(PatchId(0), MapId(0)); s]),
            output_size: size,
            unresolved: Mutex::new(unresolved),
            resolved: RwLock::new(resolved),
            tree_grid,
            locked_resolved,
            input_dimensions: Vec::new(),
        }
    }

    // Write resolved pixels from the update queue to an already write-locked `rtree` and `resolved` array.
    fn flush_resolved(
        &self,
        my_resolved_list: &mut Vec<(CoordFlat, Score)>,
        tree_grid: &TreeGrid,
        update_queue: &[([i32; 2], CoordFlat, Score)],
        is_tiling_mode: bool,
    ) {
        for (a, b, score) in update_queue.iter() {
            tree_grid.insert(a[0], a[1]);

            if is_tiling_mode {
                //if close to border add additional mirrors
                let x_l = ((self.output_size.width as f32) * TILING_BOUNDARY_PERCENTAGE) as i32;
                let x_r = self.output_size.width as i32 - x_l;
                let y_b = ((self.output_size.height as f32) * TILING_BOUNDARY_PERCENTAGE) as i32;
                let y_t = self.output_size.height as i32 - y_b;

                if a[0] < x_l {
                    tree_grid.insert(a[0] + (self.output_size.width as i32), a[1]);
                // +x
                } else if a[0] > x_r {
                    tree_grid.insert(a[0] - (self.output_size.width as i32), a[1]);
                    // -x
                }

                if a[1] < y_b {
                    tree_grid.insert(a[0], a[1] + (self.output_size.height as i32));
                // +Y
                } else if a[1] > y_t {
                    tree_grid.insert(a[0], a[1] - (self.output_size.height as i32));
                    // -Y
                }
            }
            my_resolved_list.push((*b, *score));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        my_resolved_list: &mut Vec<(CoordFlat, Score)>,
        update_coord: Coord2D,
        (example_coord, example_map_id): (Coord2D, MapId),
        example_maps: &[ImageBuffer<'_>],
        update_resolved_list: bool,
        score: Score,
        island_id: (PatchId, MapId),
        is_tiling_mode: bool,
    ) {
        let flat_coord = update_coord.to_flat(self.output_size);

        // A little cheat to avoid taking excessive locks.
        //
        // Access to `coord_map` and `color_map` is governed by values in `self.resolved`,
        // in such a way that any values in the former will not be accessed until the latter is updated.
        // Since `coord_map` and `color_map` also contain 'plain old data', we can set them directly
        // by getting the raw pointers. The subsequent access to `self.resolved` goes through a lock,
        // and ensures correct memory ordering.
        unsafe {
            self.coord_map
                .assign_at(flat_coord.0 as usize, (example_coord, example_map_id));
            self.id_map.assign_at(flat_coord.0 as usize, island_id);
        }
        self.color_map.put_pixel(
            update_coord.x,
            update_coord.y,
            *example_maps[example_map_id.0 as usize].get_pixel(example_coord.x, example_coord.y),
        );

        if update_resolved_list {
            self.flush_resolved(
                my_resolved_list,
                &self.tree_grid,
                &[(
                    [update_coord.x as i32, update_coord.y as i32],
                    flat_coord,
                    score,
                )],
                is_tiling_mode,
            );
        }
    }

    //returns flat coord
    fn pick_random_unresolved(&self, seed: u64) -> Option<CoordFlat> {
        let mut unresolved = self.unresolved.lock().unwrap();

        if unresolved.len() == 0 {
            None //return fail
        } else {
            let rand_index = Pcg32::seed_from_u64(seed).gen_range(0, unresolved.len());
            Some(unresolved.swap_remove(rand_index)) //return success
        }
    }

    fn find_k_nearest_resolved_neighs(
        &self,
        coord: Coord2D,
        k: u32,
        k_neighs_2d: &mut Vec<SignedCoord2D>,
    ) -> bool {
        self.tree_grid
            .get_k_nearest_neighbors(coord.x, coord.y, k as usize, k_neighs_2d);
        if k_neighs_2d.is_empty() {
            return false;
        }
        true
    }

    fn get_distances_to_k_neighs(&self, coord: Coord2D, k_neighs_2d: &[SignedCoord2D]) -> Vec<f64> {
        let (dimx, dimy) = (
            f64::from(self.output_size.width),
            f64::from(self.output_size.height),
        );
        let (x2, y2) = (f64::from(coord.x) / dimx, f64::from(coord.y) / dimy);
        let mut k_neighs_dist: Vec<f64> = Vec::with_capacity(k_neighs_2d.len() * 4);

        for coord in k_neighs_2d.iter() {
            let (x1, y1) = ((f64::from(coord.x)) / dimx, (f64::from(coord.y)) / dimy);
            let dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
            // Duplicate the distance for each of our 4 channels
            k_neighs_dist.extend_from_slice(&[dist, dist, dist, dist]);
        }

        //divide by avg
        let avg: f64 = k_neighs_dist.iter().sum::<f64>() / (k_neighs_dist.len() as f64);

        k_neighs_dist.iter_mut().for_each(|d| *d /= avg);
        k_neighs_dist
    }

    pub(crate) fn resolve_random_batch(
        &mut self,
        steps: usize,
        example_maps: &[ImageBuffer<'_>],
        seed: u64,
    ) {
        for i in 0..steps {
            if let Some(ref unresolved_flat) = self.pick_random_unresolved(seed + i as u64) {
                //no resolved neighs? resolve at random!
                self.resolve_at_random(
                    &mut self.resolved.write().unwrap(),
                    unresolved_flat.to_2d(self.output_size),
                    example_maps,
                    seed + i as u64 + u64::from(unresolved_flat.0),
                );
            }
        }
        self.locked_resolved += steps; //lock these pixels from being re-resolved
    }

    fn resolve_at_random(
        &self,
        my_resolved_list: &mut Vec<(CoordFlat, Score)>,
        my_coord: Coord2D,
        example_maps: &[ImageBuffer<'_>],
        seed: u64,
    ) {
        let rand_map: u32 = Pcg32::seed_from_u64(seed).gen_range(0, example_maps.len()) as u32;
        let rand_x: u32 =
            Pcg32::seed_from_u64(seed).gen_range(0, example_maps[rand_map as usize].width as u32);
        let rand_y: u32 =
            Pcg32::seed_from_u64(seed).gen_range(0, example_maps[rand_map as usize].height as u32);

        self.update(
            my_resolved_list,
            my_coord,
            (Coord2D::from(rand_x, rand_y), MapId(rand_map)),
            example_maps,
            true,
            // NOTE: giving score 0.0 which is absolutely imaginery since we're randomly
            // initializing
            Score(0.0),
            (
                PatchId(my_coord.to_flat(self.output_size).0),
                MapId(rand_map),
            ),
            false,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn find_candidates<'a>(
        &self,
        candidates_vec: &'a mut Vec<CandidateStruct>,
        unresolved_coord: Coord2D,
        k_neighs: &[SignedCoord2D],
        example_maps: &[ImageBuffer<'_>],
        valid_non_ignored_samples_mask: &[&SamplingMethod],
        m_rand: u32,
        m_seed: u64,
    ) -> &'a [CandidateStruct] {
        let mut candidate_count = 0;
        let unresolved_coord = unresolved_coord.to_signed();

        let wrap_dim = (
            self.output_size.width as i32,
            self.output_size.height as i32,
        );

        //neighborhood based candidates
        for neigh_coord in k_neighs {
            //calculate the shift between the center coord and its found neighbor
            let shift = (
                unresolved_coord.x - (*neigh_coord).x,
                unresolved_coord.y - (*neigh_coord).y,
            );

            //find center coord original location in the example map
            let n_flat_coord = neigh_coord
                .wrap(wrap_dim)
                .to_unsigned()
                .to_flat(self.output_size)
                .0 as usize;
            let (n_original_coord, _) = self.coord_map.as_ref()[n_flat_coord];
            let (n_patch_id, n_map_id) = self.id_map.as_ref()[n_flat_coord];
            //candidate coord is the original location of the neighbor + neighbor's shift to the center
            let candidate_coord = SignedCoord2D::from(
                n_original_coord.x as i32 + shift.0,
                n_original_coord.y as i32 + shift.1,
            );
            //check if the shifted coord is valid (discard if not)
            if check_coord_validity(
                candidate_coord,
                n_map_id,
                example_maps,
                valid_non_ignored_samples_mask[n_map_id.0 as usize],
            ) {
                //lets construct the full candidate pattern of neighbors identical to the center coord
                candidates_vec[candidate_count]
                    .k_neighs
                    .resize(k_neighs.len(), (SignedCoord2D::from(0, 0), MapId(0)));

                for (output, n2) in candidates_vec[candidate_count]
                    .k_neighs
                    .iter_mut()
                    .zip(k_neighs)
                {
                    let shift = (n2.x - unresolved_coord.x, n2.y - unresolved_coord.y);
                    let n2_coord = SignedCoord2D::from(
                        candidate_coord.x + shift.0,
                        candidate_coord.y + shift.1,
                    );

                    *output = (n2_coord, n_map_id)
                }
                //record the candidate info
                candidates_vec[candidate_count].coord = (candidate_coord, n_map_id);
                candidates_vec[candidate_count].id = (n_patch_id, n_map_id);
                candidate_count += 1;
            }
        }

        let mut rng = Pcg32::seed_from_u64(m_seed);
        //random candidates
        for _ in 0..m_rand {
            let rand_map = (rng.gen_range(0, example_maps.len())) as u32;
            let dims = example_maps[rand_map as usize].dimensions();
            let dims = Dims {
                width: dims.0,
                height: dims.1,
            };
            let mut rand_x: i32;
            let mut rand_y: i32;
            let mut candidate_coord;
            //generate a random valid candidate
            loop {
                rand_x = rng.gen_range(0, dims.width) as i32;
                rand_y = rng.gen_range(0, dims.height) as i32;
                candidate_coord = SignedCoord2D::from(rand_x, rand_y);
                if check_coord_validity(
                    candidate_coord,
                    MapId(rand_map),
                    example_maps,
                    valid_non_ignored_samples_mask[rand_map as usize],
                ) {
                    break;
                }
            }
            //for patch id (since we are not copying from a generated patch anymore), we take the pixel location in the example map
            let map_id = MapId(rand_map);
            let patch_id = PatchId(candidate_coord.to_unsigned().to_flat(dims).0);
            //lets construct the full neighborhood pattern
            candidates_vec[candidate_count]
                .k_neighs
                .resize(k_neighs.len(), (SignedCoord2D::from(0, 0), MapId(0)));

            for (output, n2) in candidates_vec[candidate_count]
                .k_neighs
                .iter_mut()
                .zip(k_neighs)
            {
                let shift = (unresolved_coord.x - n2.x, unresolved_coord.y - n2.y);
                let n2_coord =
                    SignedCoord2D::from(candidate_coord.x + shift.0, candidate_coord.y + shift.1);

                *output = (n2_coord, map_id)
            }

            //record the candidate info
            candidates_vec[candidate_count].coord = (candidate_coord, map_id);
            candidates_vec[candidate_count].id = (patch_id, map_id);
            candidate_count += 1;
        }

        &candidates_vec[0..candidate_count]
    }

    /// Returns an image of Ids for visualizing the 'copy islands' and map ids of those islands
    pub fn get_id_maps(&self) -> [image::RgbaImage; 2] {
        //init empty image
        let mut map_id_map = image::RgbaImage::new(self.output_size.width, self.output_size.height);
        let mut patch_id_map =
            image::RgbaImage::new(self.output_size.width, self.output_size.height);
        //populate the image with colors
        for (i, (patch_id, map_id)) in self.id_map.as_ref().iter().enumerate() {
            //get 2d coord
            let coord = CoordFlat(i as u32).to_2d(self.output_size);
            //get random color based on id
            let color: image::Rgba<u8> = image::Rgba([
                Pcg32::seed_from_u64(u64::from(patch_id.0)).gen_range(0, 255),
                Pcg32::seed_from_u64(u64::from((patch_id.0) * 5 + 21)).gen_range(0, 255),
                Pcg32::seed_from_u64(u64::from((patch_id.0) / 4 + 12)).gen_range(0, 255),
                255,
            ]);
            //write image
            patch_id_map.put_pixel(coord.x, coord.y, color);
            //get random color based on id
            let color: image::Rgba<u8> = image::Rgba([
                Pcg32::seed_from_u64(u64::from(map_id.0) * 200).gen_range(0, 255),
                Pcg32::seed_from_u64(u64::from((map_id.0) * 5 + 341)).gen_range(0, 255),
                Pcg32::seed_from_u64(u64::from((map_id.0) * 1200 - 35412)).gen_range(0, 255),
                255,
            ]);
            map_id_map.put_pixel(coord.x, coord.y, color);
        }
        [patch_id_map, map_id_map]
    }

    pub fn get_uncertainty_map(&self) -> image::RgbaImage {
        let mut uncertainty_map =
            image::RgbaImage::new(self.output_size.width, self.output_size.height);

        for (flat_coord, score) in self.resolved.read().unwrap().iter() {
            //get coord
            let coord = flat_coord.to_2d(self.output_size);
            //get value normalized
            let normalized_score = (score.0.min(1.0) * 255.0) as u8;

            let color: image::Rgba<u8> =
                image::Rgba([normalized_score, 255 - normalized_score, 0, 255]);

            //write image
            uncertainty_map.put_pixel(coord.x, coord.y, color);
        }

        uncertainty_map
    }

    pub fn get_coord_transform(&self) -> CoordinateTransform {
        // init empty 32bit image
        let coord_map = self.coord_map.as_ref();

        let mut buffer: Vec<u32> = Vec::new();

        // presize the vector for our final size
        buffer.resize(coord_map.len() * 3, 0);

        //populate the image with colors
        for (i, (coord, map_id)) in self.coord_map.as_ref().iter().enumerate() {
            let b = map_id.0;

            //record the color
            let ind = i * 3;
            let color = &mut buffer[ind..ind + 3];

            color[0] = coord.x;
            color[1] = coord.y;
            color[2] = b;
        }

        let original_maps = self.input_dimensions.clone();

        CoordinateTransform {
            buffer,
            output_size: Dims::new(self.output_size.width, self.output_size.height),
            original_maps,
        }
    }

    //replace every resolved pixel with a pixel from a new level
    fn next_pyramid_level(&mut self, example_maps: &[ImageBuffer<'_>]) {
        for (coord_flat, _) in self.resolved.read().unwrap().iter() {
            let resolved_2d = coord_flat.to_2d(self.output_size);
            let (example_map_coord, example_map_id) =
                self.coord_map.as_ref()[coord_flat.0 as usize]; //so where the current pixel came from

            self.color_map.put_pixel(
                resolved_2d.x,
                resolved_2d.y,
                *example_maps[example_map_id.0 as usize]
                    .get_pixel(example_map_coord.x, example_map_coord.y),
            );
        }
    }

    pub(crate) fn resolve(
        &mut self,
        params: &GeneratorParams,
        example_maps_pyramid: &[ImagePyramid],
        mut progress: Option<Box<dyn crate::GeneratorProgress>>,
        guides_pyramid: &Option<GuidesPyramidStruct>,
        valid_samples: &[SamplingMethod],
    ) {
        let total_pixels_to_resolve = self.unresolved.lock().unwrap().len();

        // Currently we do not give progress for wasm
        #[cfg(not(target_arch = "wasm32"))]
        let mut total_processed_pixels = 0;

        let mut pyramid_level = 0;

        let valid_non_ignored_samples: Vec<&SamplingMethod> = valid_samples[..]
            .iter()
            .filter(|s| !s.is_ignore())
            .collect();

        // Get the dimensions for each input example, this is only used when
        // saving a coordinate transform, so that the transform can be repeated
        // with different inputs that can be resized to avoid various problems
        self.input_dimensions = example_maps_pyramid
            .iter()
            .map(|ip| {
                let original = ip.bottom();
                Dims {
                    width: original.width(),
                    height: original.height(),
                }
            })
            .collect();

        let stage_pixels_to_resolve = |p_stage: i32| {
            (params.p.powf(p_stage as f32) * (total_pixels_to_resolve as f32)) as usize
        };

        let is_tiling_mode = params.tiling_mode;

        let cauchy_precomputed = PrerenderedU8Function::new(|a, b| {
            metric_cauchy(a, b, params.cauchy_dispersion * params.cauchy_dispersion)
        });
        let l2_precomputed = PrerenderedU8Function::new(metric_l2);
        let max_workers = params.max_thread_count;
        // Use a single R*-tree initially, and fan out to a grid of them later?
        let mut has_fanned_out = false;

        {
            // now that we have all of the parameters we can setup our initial tree grid
            let tile_adjusted_width =
                (self.output_size.width as f32 * (1.0 + TILING_BOUNDARY_PERCENTAGE * 2.0)) as u32
                    + 1;
            let tile_adjusted_height =
                (self.output_size.height as f32 * (1.0 + TILING_BOUNDARY_PERCENTAGE * 2.0)) as u32
                    + 1;
            self.tree_grid = TreeGrid::new(
                tile_adjusted_width,
                tile_adjusted_height,
                max(tile_adjusted_width, tile_adjusted_height),
                (self.output_size.width as f32 * TILING_BOUNDARY_PERCENTAGE) as u32 + 1,
                (self.output_size.height as f32 * TILING_BOUNDARY_PERCENTAGE) as u32 + 1,
            );
            // if we already have resolved pixels from an inpaint or multiexample add them to this tree grid
            let resolved_queue = &mut self.resolved.write().unwrap();
            let pixels_to_update: Vec<([i32; 2], CoordFlat, Score)> = resolved_queue
                .drain(..)
                .map(|a| {
                    let coord_2d = a.0.to_2d(self.output_size);
                    ([coord_2d.x as i32, coord_2d.y as i32], a.0, a.1)
                })
                .collect();
            self.flush_resolved(
                resolved_queue,
                &self.tree_grid,
                &pixels_to_update[..],
                is_tiling_mode,
            );
        }

        for p_stage in (0..=params.p_stages).rev() {
            //get maps from current pyramid level (for now it will be p-stage dependant)
            let example_maps = get_single_example_level(
                example_maps_pyramid,
                valid_samples,
                pyramid_level as usize,
            );
            let guides = get_single_guide_level(guides_pyramid, pyramid_level as usize);

            //update pyramid level
            if pyramid_level > 0 {
                self.next_pyramid_level(&example_maps);
            }
            pyramid_level += 1;
            pyramid_level = pyramid_level.min(params.p_stages - 1); //dont go beyond

            //get seed
            let p_stage_seed: u64 =
                u64::from(Pcg32::seed_from_u64(params.seed + p_stage as u64).gen::<u32>());

            //how many pixels do we need to resolve in this stage
            let pixels_to_resolve = stage_pixels_to_resolve(p_stage);
            let redo_count = self.resolved.get_mut().unwrap().len() - self.locked_resolved;

            // Start with serial execution for the first few pixels, then go wide
            let n_workers = if redo_count < 1000 { 1 } else { max_workers };
            if !has_fanned_out && n_workers > 1 {
                has_fanned_out = true;
                let tile_adjusted_width = (self.output_size.width as f32
                    * (1.0 + TILING_BOUNDARY_PERCENTAGE * 2.0))
                    as u32
                    + 1;
                let tile_adjusted_height = (self.output_size.height as f32
                    * (1.0 + TILING_BOUNDARY_PERCENTAGE * 2.0))
                    as u32
                    + 1;
                // heuristic: pick a cell size so that the expected number of resolved points in any cell is 4 * k
                // this seems to be a safe overestimate
                let grid_cell_size =
                    ((params.nearest_neighbors * self.output_size.width * self.output_size.height
                        / redo_count as u32) as f64)
                        .sqrt() as u32
                        * 2
                        + 1;
                let new_tree_grid = TreeGrid::new(
                    tile_adjusted_width,
                    tile_adjusted_height,
                    grid_cell_size,
                    (self.output_size.width as f32 * TILING_BOUNDARY_PERCENTAGE) as u32 + 1,
                    (self.output_size.height as f32 * TILING_BOUNDARY_PERCENTAGE) as u32 + 1,
                );
                self.tree_grid.clone_into_new_tree_grid(&new_tree_grid);
                self.tree_grid = new_tree_grid;
            }

            //calculate the guidance alpha
            let adaptive_alpha = if guides.is_some() && p_stage > 0 {
                let total_resolved = self.resolved.read().unwrap().len() as f32;
                (params.alpha * (1.0 - (total_resolved / (total_pixels_to_resolve as f32))))
                    .powf(3.0)
            } else {
                0.0 //only care for content, not guidance
            };

            let guide_cost_precomputed =
                PrerenderedU8Function::new(|a, b| adaptive_alpha * l2_precomputed.get(a, b));

            let my_inverse_alpha_cost_precomputed = PrerenderedU8Function::new(|a, b| {
                (1.0 - adaptive_alpha) * cauchy_precomputed.get(a, b)
            });

            // Keep track of how many items have been processed. Goes up to `pixels_to_resolve`
            let processed_pixel_count = AtomicUsize::new(0);
            let remaining_threads = AtomicUsize::new(n_workers);

            let mut pixels_resolved_this_stage: Vec<Mutex<Vec<(CoordFlat, Score)>>> = Vec::new();
            pixels_resolved_this_stage.resize_with(n_workers, || Mutex::new(Vec::new()));
            let thread_counter = AtomicUsize::new(0);

            let worker_fn = || {
                let mut candidates: Vec<CandidateStruct> = Vec::new();
                let mut my_pattern: ColorPattern = ColorPattern::new();
                let mut k_neighs: Vec<SignedCoord2D> =
                    Vec::with_capacity(params.nearest_neighbors as usize);

                let max_candidate_count =
                    params.nearest_neighbors as usize + params.random_sample_locations as usize;

                let my_thread_id = thread_counter.fetch_add(1, Ordering::Relaxed);
                let mut my_resolved_list = pixels_resolved_this_stage[my_thread_id].lock().unwrap();

                candidates.resize(max_candidate_count, CandidateStruct::default());

                //alloc storage for our guides (regardless of whether we have them or not)
                let mut my_guide_pattern: ColorPattern = ColorPattern::new();

                let out_color_map = &[ImageBuffer::from(self.color_map.as_ref())];

                loop {
                    // Get the next work item
                    let i = processed_pixel_count.fetch_add(1, Ordering::Relaxed);

                    let update_resolved_list: bool;

                    if i >= pixels_to_resolve {
                        // We've processed everything, so finish the worker
                        break;
                    }

                    let loop_seed = p_stage_seed + i as u64;

                    // 1. Get a pixel to resolve. Check if we have already resolved pixel i; if yes, resolve again; if no, pick a new one
                    let next_unresolved = if i < redo_count {
                        update_resolved_list = false;
                        self.resolved.read().unwrap()[i + self.locked_resolved].0
                    } else {
                        update_resolved_list = true;
                        if let Some(pixel) = self.pick_random_unresolved(loop_seed) {
                            pixel
                        } else {
                            break;
                        }
                    };

                    let unresolved_2d = next_unresolved.to_2d(self.output_size);

                    // Clear previously found candidate neighbors
                    for cand in candidates.iter_mut() {
                        cand.clear();
                    }
                    k_neighs.clear();

                    // 2. find K nearest resolved neighs
                    if self.find_k_nearest_resolved_neighs(
                        unresolved_2d,
                        params.nearest_neighbors,
                        &mut k_neighs,
                    ) {
                        //2.1 get distances to the pattern of neighbors
                        let k_neighs_dist =
                            self.get_distances_to_k_neighs(unresolved_2d, &k_neighs);
                        let k_neighs_w_map_id =
                            k_neighs.iter().map(|a| (*a, MapId(0))).collect::<Vec<_>>();

                        // 3. find candidate for each resolved neighs + m random locations
                        let candidates: &[CandidateStruct] = self.find_candidates(
                            &mut candidates,
                            unresolved_2d,
                            &k_neighs,
                            &example_maps,
                            &valid_non_ignored_samples,
                            params.random_sample_locations as u32,
                            loop_seed + 1,
                        );

                        k_neighs_to_precomputed_reference_pattern(
                            &k_neighs_w_map_id, //feed into the function with always 0 index of the sample map
                            image::Rgba([0, 0, 0, 255]),
                            out_color_map,
                            &mut my_pattern,
                            is_tiling_mode,
                        );

                        // 3.2 get pattern for guide map if we have them
                        let (my_cost, guide_cost) = if let Some(ref in_guides) = guides {
                            //get example pattern to compare to
                            k_neighs_to_precomputed_reference_pattern(
                                &k_neighs_w_map_id,
                                image::Rgba([0, 0, 0, 255]),
                                &[in_guides.target_guide.clone()],
                                &mut my_guide_pattern,
                                is_tiling_mode,
                            );

                            (
                                &my_inverse_alpha_cost_precomputed,
                                Some(&guide_cost_precomputed),
                            )
                        } else {
                            (&cauchy_precomputed, None)
                        };

                        // 4. find best match based on the candidate patterns
                        let (best_match, score) = find_best_match(
                            image::Rgba([0, 0, 0, 255]),
                            &example_maps,
                            &guides,
                            candidates,
                            &my_pattern,
                            &my_guide_pattern,
                            &k_neighs_dist,
                            my_cost,
                            guide_cost,
                        );

                        let best_match_coord = best_match.coord.0.to_unsigned();
                        let best_match_map_id = best_match.coord.1;

                        // 5. resolve our pixel
                        self.update(
                            &mut my_resolved_list,
                            unresolved_2d,
                            (best_match_coord, best_match_map_id),
                            &example_maps,
                            update_resolved_list,
                            score,
                            best_match.id,
                            is_tiling_mode,
                        );
                    } else {
                        //no resolved neighs? resolve at random!
                        self.resolve_at_random(
                            &mut my_resolved_list,
                            unresolved_2d,
                            &example_maps,
                            p_stage_seed,
                        );
                    }
                }
                remaining_threads.fetch_sub(1, Ordering::Relaxed);
            };

            // for WASM we do not have threads and crossbeam panics,
            // so let's just run the worker function directly and don't give progress
            #[cfg(target_arch = "wasm32")]
            (worker_fn)();

            #[cfg(not(target_arch = "wasm32"))]
            {
                let actual_total_pixels_to_resolve: usize =
                    (0..=params.p_stages).map(stage_pixels_to_resolve).sum();

                crossbeam_utils::thread::scope(|scope| {
                    for _ in 0..n_workers {
                        scope.spawn(|_| (worker_fn)());
                    }

                    if let Some(ref mut progress) = progress {
                        let mut last_pcnt = 0;

                        loop {
                            let stage_progress = processed_pixel_count.load(Ordering::Relaxed);

                            if remaining_threads.load(Ordering::Relaxed) == 0 {
                                break;
                            }

                            let pcnt = ((total_processed_pixels + stage_progress) as f32
                                / actual_total_pixels_to_resolve as f32
                                * 100f32)
                                .round() as u32;

                            if pcnt != last_pcnt {
                                progress.update(crate::ProgressUpdate {
                                    image: self.color_map.as_ref(),
                                    total: crate::ProgressStat {
                                        total: actual_total_pixels_to_resolve,
                                        current: total_processed_pixels + stage_progress,
                                    },
                                    stage: crate::ProgressStat {
                                        total: pixels_to_resolve,
                                        current: stage_progress,
                                    },
                                });

                                last_pcnt = pcnt;
                            }
                        }

                        total_processed_pixels += pixels_to_resolve;
                    }
                })
                .unwrap();
            }

            {
                // append all per-thread resolved lists to the global list
                let mut resolved = self.resolved.write().unwrap();
                for thread_resolved in pixels_resolved_this_stage {
                    resolved.append(&mut thread_resolved.into_inner().unwrap());
                }
            }
        }
    }
}

#[inline]
fn metric_cauchy(a: u8, b: u8, sig2: f32) -> f32 {
    let mut x2 = (f32::from(a) - f32::from(b)) / 255.0; //normalize the colors to be between 0-1
    x2 = x2 * x2;
    (1.0 + x2 / sig2).ln()
}

#[inline]
fn metric_l2(a: u8, b: u8) -> f32 {
    let x = (f32::from(a) - f32::from(b)) / 255.0;
    x * x
}

#[inline]
fn get_color_of_neighbor(
    outside_color: image::Rgba<u8>,
    source_maps: &[ImageBuffer<'_>],
    n_coord: SignedCoord2D,
    n_map: MapId,
    neighbor_color: &mut [u8],
    is_wrap_mode: bool,
    wrap_dim: (i32, i32),
) {
    let coord = if is_wrap_mode {
        n_coord.wrap(wrap_dim)
    } else {
        n_coord
    };

    //check if he haven't gone outside the possible bounds
    if source_maps[n_map.0 as usize].is_in_bounds(coord) {
        neighbor_color.copy_from_slice(
            &(source_maps[n_map.0 as usize])
                .get_pixel(coord.x as u32, coord.y as u32)
                .0[..4],
        );
    } else {
        // if we have gone out of bounds, then just fill as outside color
        neighbor_color.copy_from_slice(&outside_color.0[..]);
    }
}

fn k_neighs_to_precomputed_reference_pattern(
    k_neighs: &[(SignedCoord2D, MapId)],
    outside_color: image::Rgba<u8>,
    source_maps: &[ImageBuffer<'_>],
    pattern: &mut ColorPattern,
    is_wrap_mode: bool,
) {
    pattern.0.resize(k_neighs.len() * 4, 0);
    let mut i = 0;

    let wrap_dim = (
        source_maps[0].dimensions().0 as i32,
        source_maps[0].dimensions().1 as i32,
    );

    for (n_coord, n_map) in k_neighs {
        let end = i + 4;

        get_color_of_neighbor(
            outside_color,
            source_maps,
            *n_coord,
            *n_map,
            &mut (pattern.0[i..end]),
            is_wrap_mode,
            wrap_dim,
        );

        i = end;
    }
}

#[allow(clippy::too_many_arguments)]
fn find_best_match<'a>(
    outside_color: image::Rgba<u8>,
    source_maps: &[ImageBuffer<'_>],
    guides: &Option<GuidesStruct<'_>>,
    candidates: &'a [CandidateStruct],
    my_precomputed_pattern: &ColorPattern,
    my_precomputed_guide_pattern: &ColorPattern,
    k_distances: &[f64], //weight by distance
    my_cost: &PrerenderedU8Function,
    guide_cost: Option<&PrerenderedU8Function>,
) -> (&'a CandidateStruct, Score) {
    let mut best_match = 0;
    let mut lowest_cost = std::f32::MAX;

    let distance_gaussians: Vec<f32> = k_distances
        .iter()
        .copied()
        .map(|d| f64::exp(-1.0f64 * d))
        .map(|d| d as f32)
        .collect();

    for (i, cand) in candidates.iter().enumerate() {
        if let Some(cost) = better_match(
            &cand.k_neighs,
            outside_color,
            source_maps,
            guides,
            my_precomputed_pattern,
            my_precomputed_guide_pattern,
            distance_gaussians.as_slice(),
            my_cost,
            guide_cost,
            lowest_cost,
        ) {
            lowest_cost = cost;
            best_match = i;
        }
    }

    (&candidates[best_match], Score(lowest_cost))
}

#[allow(clippy::too_many_arguments)]
fn better_match(
    k_neighs: &[(SignedCoord2D, MapId)],
    outside_color: image::Rgba<u8>,
    source_maps: &[ImageBuffer<'_>],
    guides: &Option<GuidesStruct<'_>>,
    my_precomputed_pattern: &ColorPattern,
    my_precomputed_guide_pattern: &ColorPattern,
    distance_gaussians: &[f32], //weight by distance
    my_cost: &PrerenderedU8Function,
    guide_cost: Option<&PrerenderedU8Function>,
    current_best: f32,
) -> Option<f32> {
    let mut score: f32 = 0.0; //minimize score

    let mut i = 0;
    let mut next_pixel = [0; 4];
    let mut next_pixel_score: f32;
    for (n_coord, n_map) in k_neighs {
        next_pixel_score = 0.0;
        let end = i + 4;

        //check if he haven't gone outside the possible bounds
        get_color_of_neighbor(
            outside_color,
            source_maps,
            *n_coord,
            *n_map,
            &mut next_pixel,
            false,
            (0, 0),
        );

        for (channel_n, &channel) in next_pixel.iter().enumerate() {
            next_pixel_score += my_cost.get(my_precomputed_pattern.0[i + channel_n], channel);
        }

        if let Some(guide_cost) = guide_cost {
            let example_guides = &(guides.as_ref().unwrap().example_guides);
            get_color_of_neighbor(
                outside_color,
                example_guides,
                *n_coord,
                *n_map,
                &mut next_pixel,
                false,
                (0, 0),
            );

            for (channel_n, &channel) in next_pixel.iter().enumerate() {
                next_pixel_score +=
                    guide_cost.get(my_precomputed_guide_pattern.0[i + channel_n], channel);
            }
        }
        score += next_pixel_score * distance_gaussians[i];
        if score >= current_best {
            return None;
        }
        i = end;
    }

    Some(score)
}

struct PrerenderedU8Function {
    data: Vec<f32>,
}

impl PrerenderedU8Function {
    pub fn new<F: Fn(u8, u8) -> f32>(function: F) -> Self {
        let mut data = vec![0f32; 65536];

        for a in 0..=255u8 {
            for b in 0..=255u8 {
                data[a as usize * 256usize + b as usize] = function(a, b);
            }
        }

        Self { data }
    }

    #[inline]
    pub fn get(&self, a: u8, b: u8) -> f32 {
        self.data[a as usize * 256usize + b as usize]
    }
}

struct TreeGrid {
    grid_width: u32,
    grid_height: u32,
    offset_x: i32,
    offset_y: i32,
    chunk_size: u32,
    rtrees: Vec<RwLock<RTree<[i32; 2]>>>,
}

// This is a grid of rtrees
// The idea is that most pixels after the first couple steps will have their neighbors close by
impl TreeGrid {
    pub fn new(width: u32, height: u32, chunk_size: u32, offset_x: u32, offset_y: u32) -> Self {
        let mut rtrees: Vec<RwLock<RTree<[i32; 2]>>> = Vec::new();
        let grid_width = max((width + chunk_size - 1) / chunk_size, 1);
        let grid_height = max((height + chunk_size - 1) / chunk_size, 1);
        rtrees.resize_with((grid_width * grid_height) as usize, || {
            RwLock::new(RTree::new())
        });
        Self {
            rtrees,
            grid_width,
            grid_height,
            offset_x: offset_x as i32,
            offset_y: offset_y as i32,
            chunk_size,
        }
    }

    #[inline]
    fn get_tree_index(&self, x: u32, y: u32) -> usize {
        (x * self.grid_height + y) as usize
    }

    pub fn insert(&self, x: i32, y: i32) {
        let my_tree_index = self.get_tree_index(
            ((x + self.offset_x) as u32) / self.chunk_size,
            ((y + self.offset_y) as u32) / self.chunk_size,
        );
        self.rtrees[my_tree_index].write().unwrap().insert([x, y]);
    }

    pub fn clone_into_new_tree_grid(&self, other: &Self) {
        for tree in &self.rtrees {
            for coord in tree.read().unwrap().iter() {
                other.insert((*coord)[0], (*coord)[1]);
            }
        }
    }

    pub fn get_k_nearest_neighbors(
        &self,
        x: u32,
        y: u32,
        k: usize,
        result: &mut Vec<SignedCoord2D>,
    ) {
        let offset_x = x as i32 + self.offset_x;
        let offset_y = y as i32 + self.offset_y;

        let chunk_x = offset_x / self.chunk_size as i32;
        let chunk_y = offset_y / self.chunk_size as i32;

        struct ChunkSearchInfo {
            x: i32,
            y: i32,
            center: bool,
            closest_point_on_boundary_x: i64,
            closest_point_on_boundary_y: i64,
        }

        // Assume that all k nearest neighbors are in these cells
        // it looks like we are rarely wrong once enough pixels are filled in
        let places_to_look = [
            ChunkSearchInfo {
                x: chunk_x,
                y: chunk_y,
                center: true,
                closest_point_on_boundary_x: 0,
                closest_point_on_boundary_y: 0,
            },
            ChunkSearchInfo {
                x: chunk_x + 1,
                y: chunk_y,
                center: false,
                closest_point_on_boundary_x: ((chunk_x + 1) * self.chunk_size as i32
                    - self.offset_x) as i64,
                closest_point_on_boundary_y: y as i64,
            },
            ChunkSearchInfo {
                x: chunk_x - 1,
                y: chunk_y,
                center: false,
                closest_point_on_boundary_x: (chunk_x * self.chunk_size as i32 - self.offset_x)
                    as i64,
                closest_point_on_boundary_y: y as i64,
            },
            ChunkSearchInfo {
                x: chunk_x,
                y: chunk_y - 1,
                center: false,
                closest_point_on_boundary_x: x as i64,
                closest_point_on_boundary_y: (chunk_y * self.chunk_size as i32 - self.offset_y)
                    as i64,
            },
            ChunkSearchInfo {
                x: chunk_x,
                y: chunk_y + 1,
                center: false,
                closest_point_on_boundary_x: x as i64,
                closest_point_on_boundary_y: ((chunk_y + 1) * self.chunk_size as i32
                    - self.offset_y) as i64,
            },
            ChunkSearchInfo {
                x: chunk_x + 1,
                y: chunk_y + 1,
                center: false,
                closest_point_on_boundary_x: ((chunk_x + 1) * self.chunk_size as i32
                    - self.offset_x) as i64,
                closest_point_on_boundary_y: ((chunk_y + 1) * self.chunk_size as i32
                    - self.offset_y) as i64,
            },
            ChunkSearchInfo {
                x: chunk_x - 1,
                y: chunk_y + 1,
                center: false,
                closest_point_on_boundary_x: (chunk_x * self.chunk_size as i32 - self.offset_x)
                    as i64,
                closest_point_on_boundary_y: ((chunk_y + 1) * self.chunk_size as i32
                    - self.offset_y) as i64,
            },
            ChunkSearchInfo {
                x: chunk_x + 1,
                y: chunk_y - 1,
                center: false,
                closest_point_on_boundary_x: ((chunk_x + 1) * self.chunk_size as i32
                    - self.offset_x) as i64,
                closest_point_on_boundary_y: (chunk_y * self.chunk_size as i32 - self.offset_y)
                    as i64,
            },
            ChunkSearchInfo {
                x: chunk_x - 1,
                y: chunk_y - 1,
                center: false,
                closest_point_on_boundary_x: (chunk_x * self.chunk_size as i32 - self.offset_x)
                    as i64,
                closest_point_on_boundary_y: (chunk_y * self.chunk_size as i32 - self.offset_y)
                    as i64,
            },
        ];
        // Note locking all of them at different times seems to be the best way
        // Naively trying to lock all at once could easily result in deadlocks
        let mut tmp_result: Vec<(i32, i32, i64)> = Vec::with_capacity(k * 9);
        result.clear();
        result.reserve(k);

        // an upper bound is good enough here
        let mut upper_bound_kth_best_squared_distance = i64::max_value();
        for place_to_look in places_to_look.iter() {
            if place_to_look.x >= 0
                && place_to_look.x < self.grid_width as i32
                && place_to_look.y >= 0
                && place_to_look.y < self.grid_height as i32
            {
                let is_center = place_to_look.center;

                // a tiny optimization to help us throw far away neighbors
                // saves us a decent amount of reads
                if !is_center {
                    let squared_distance_to_closest_possible_point_on_chunk = (x as i64
                        - place_to_look.closest_point_on_boundary_x)
                        * (x as i64 - place_to_look.closest_point_on_boundary_x)
                        + (y as i64 - place_to_look.closest_point_on_boundary_y)
                            * (y as i64 - place_to_look.closest_point_on_boundary_y);

                    if squared_distance_to_closest_possible_point_on_chunk
                        > upper_bound_kth_best_squared_distance
                    {
                        continue;
                    }
                }

                let my_tree_index =
                    self.get_tree_index(place_to_look.x as u32, place_to_look.y as u32);
                let my_rtree = &self.rtrees[my_tree_index];
                tmp_result.extend(
                    my_rtree
                        .read()
                        .unwrap()
                        .nearest_neighbor_iter(&[x as i32, y as i32])
                        .take(k)
                        .map(|a| {
                            (
                                (*a)[0],
                                (*a)[1],
                                ((*a)[0] as i64 - x as i64) * ((*a)[0] as i64 - x as i64)
                                    + ((*a)[1] as i64 - y as i64) * ((*a)[1] as i64 - y as i64),
                            )
                        }),
                );

                // this isn't really the kth best distance but it's an okay approximation
                if tmp_result.len() >= k {
                    let furthest_dist_for_chunk = tmp_result[tmp_result.len() - 1].2;
                    if furthest_dist_for_chunk < upper_bound_kth_best_squared_distance {
                        upper_bound_kth_best_squared_distance = furthest_dist_for_chunk;
                    }
                }
            }
        }
        tmp_result.sort_by_key(|k| k.2);
        result.extend(
            tmp_result
                .iter()
                .take(k)
                .map(|a| SignedCoord2D::from(a.0, a.1)),
        );
    }
}

#[inline]
fn check_coord_validity(
    coord: SignedCoord2D,
    map_id: MapId,
    example_maps: &[ImageBuffer<'_>],
    mask: &SamplingMethod,
) -> bool {
    if !example_maps[map_id.0 as usize].is_in_bounds(coord) {
        return false;
    }

    match mask {
        SamplingMethod::All => true,
        SamplingMethod::Image(ref img) => img[(coord.x as u32, coord.y as u32)][0] != 0,
        SamplingMethod::Ignore => unreachable!(),
    }
}

//get all the example images from a single pyramid level
fn get_single_example_level<'a>(
    example_maps_pyramid: &'a [ImagePyramid],
    valid_samples_mask: &[SamplingMethod],
    pyramid_level: usize,
) -> Vec<ImageBuffer<'a>> {
    example_maps_pyramid
        .iter()
        .enumerate()
        .filter(|&(i, _)| !valid_samples_mask[i].is_ignore())
        .map(|(_, a)| ImageBuffer::from(&a.pyramid[pyramid_level]))
        .collect()
}

//get all the guide images from a single pyramid level
fn get_single_guide_level(
    guides_pyramid: &Option<GuidesPyramidStruct>,
    pyramid_level: usize,
) -> Option<GuidesStruct<'_>> {
    guides_pyramid
        .as_ref()
        .map(|guides_pyr| guides_pyr.to_guides_struct(pyramid_level))
}
