use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;
use rstar::RTree;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use crate::{img_pyramid::*, SamplingMethod};

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
    /// The percentage of pixels to be backtracked during each p_stage. Range (0,1).
    pub(crate) p: f32,
    /// Controls the number of backtracking stages. Backtracking prevents 'garbage' generation
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
    fn wrap(self, (dimx, dimy): (i32, i32)) -> SignedCoord2D {
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

    fn to_flat(self, dims: (u32, u32)) -> CoordFlat {
        CoordFlat(dims.0 * self.y + self.x)
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
    fn to_2d(self, dims: (u32, u32)) -> Coord2D {
        let y = self.0 / dims.0;
        let x = self.0 - y * dims.0;
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
    pub(crate) color_map: image::RgbaImage,
    coord_map: Vec<(Coord2D, MapId)>, //list of samples coordinates from example map
    id_map: Vec<(PatchId, MapId)>,    // list of all id maps of our generated image
    pub(crate) output_size: (u32, u32), // size of the generated image
    unresolved: Mutex<Vec<CoordFlat>>, //for us to pick from
    resolved: RwLock<Vec<(CoordFlat, Score)>>, //a list of resolved coordinates in our canvas and their scores
    rtree: RwLock<RTree<[i32; 2]>>,            //R* tree
    update_queue: Mutex<Vec<([i32; 2], CoordFlat, Score)>>,
    locked_resolved: usize, //used for inpainting, to not backtrack these pixels
}

impl Generator {
    pub(crate) fn new(size: (u32, u32)) -> Self {
        let s = (size.0 as usize) * (size.1 as usize);
        let unresolved: Vec<CoordFlat> = (0..(s as u32)).map(CoordFlat).collect();
        Self {
            color_map: image::RgbaImage::new(size.0, size.1),
            coord_map: vec![(Coord2D::from(0, 0), MapId(0)); s],
            id_map: vec![(PatchId(0), MapId(0)); s],
            output_size: size,
            unresolved: Mutex::new(unresolved),
            resolved: RwLock::new(Vec::new()),
            rtree: RwLock::new(RTree::new()),
            update_queue: Mutex::new(Vec::new()),
            locked_resolved: 0,
        }
    }

    pub(crate) fn new_from_inpaint(
        size: (u32, u32),
        inpaint_map: image::RgbaImage,
        color_map: image::RgbaImage,
        color_map_index: usize,
    ) -> Self {
        let inpaint_map = if inpaint_map.width() != size.0 || inpaint_map.height() != size.1 {
            image::imageops::resize(&inpaint_map, size.0, size.1, image::imageops::Triangle)
        } else {
            inpaint_map
        };

        let color_map = if color_map.width() != size.0 || color_map.height() != size.1 {
            image::imageops::resize(&color_map, size.0, size.1, image::imageops::Triangle)
        } else {
            color_map
        };

        //
        let s = (size.0 as usize) * (size.1 as usize);
        let mut unresolved: Vec<CoordFlat> = Vec::new();
        let mut resolved: Vec<(CoordFlat, Score)> = Vec::new();
        let mut coord_map = vec![(Coord2D::from(0, 0), MapId(0)); s];
        let mut rtree = RTree::new();
        //populate resolved, unresolved and coord map
        for (i, pixel) in inpaint_map.clone().pixels().enumerate() {
            if pixel[0] < 255 {
                unresolved.push(CoordFlat(i as u32));
            } else {
                resolved.push((CoordFlat(i as u32), Score(0.0)));
                let coord = CoordFlat(i as u32).to_2d(size);
                coord_map[i] = (coord, MapId(color_map_index as u32)); //this absolutely requires the input image and output image to be the same size!!!!
                rtree.insert([coord.x as i32, coord.y as i32]);
            }
        }

        let locked_resolved = resolved.len();
        Self {
            color_map: color_map.clone(),
            coord_map,
            id_map: vec![(PatchId(0), MapId(0)); s],
            output_size: size,
            unresolved: Mutex::new(unresolved),
            resolved: RwLock::new(resolved),
            rtree: RwLock::new(rtree),
            update_queue: Mutex::new(Vec::new()),
            locked_resolved,
        }
    }

    // Write resolved pixels from the update queue to an already write-locked `rtree` and `resolved` array.
    fn flush_resolved(
        &self,
        rtree: &mut RTree<[i32; 2]>,
        update_queue: &[([i32; 2], CoordFlat, Score)],
        is_tiling_mode: bool,
    ) {
        let mut resolved = self.resolved.write().unwrap();

        for (a, b, score) in update_queue.iter() {
            rtree.insert(*a);

            if is_tiling_mode {
                //if close to border add additional mirrors
                let x_l = ((self.output_size.0 as f32) * 0.05) as i32;
                let x_r = self.output_size.0 as i32 - x_l;
                let y_b = ((self.output_size.1 as f32) * 0.05) as i32;
                let y_t = self.output_size.1 as i32 - y_b;

                if a[0] < x_l {
                    rtree.insert([a[0] + (self.output_size.0 as i32), a[1]]); // +x
                } else if a[0] > x_r {
                    rtree.insert([a[0] - (self.output_size.0 as i32), a[1]]); // -x
                }

                if a[1] < y_b {
                    rtree.insert([a[0], a[1] + (self.output_size.1 as i32)]); // +Y
                } else if a[1] > y_t {
                    rtree.insert([a[0], a[1] - (self.output_size.1 as i32)]); // -Y
                }
            }
            resolved.push((*b, *score));
        }
    }

    fn force_flush_resolved(&self, is_tiling_mode: bool) {
        self.flush_resolved(
            &mut *self.rtree.write().unwrap(),
            &self
                .update_queue
                .lock()
                .unwrap()
                .drain(..)
                .collect::<Vec<_>>(),
            is_tiling_mode,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
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
        #[allow(clippy::cast_ref_to_mut)]
        unsafe {
            *(self.coord_map.as_ptr() as *mut (Coord2D, MapId)).add(flat_coord.0 as usize) =
                (example_coord, example_map_id);

            *(self.id_map.as_ptr() as *mut (PatchId, MapId)).add(flat_coord.0 as usize) = island_id;

            *(self.color_map.get_pixel(update_coord.x, update_coord.y) as *const image::Rgba<u8>
                as *mut image::Rgba<u8>) = *example_maps[example_map_id.0 as usize]
                .get_pixel(example_coord.x, example_coord.y);
        }

        if update_resolved_list {
            const FORCE_FLUSH_THRESHOLD: usize = 32;

            let force_flush_items: Option<Vec<_>> = {
                let mut update_queue = self.update_queue.lock().unwrap();

                // Don't immediately resolve the pixel. Instead, add it to a list, to be resolved at the right time.
                update_queue.push((
                    [update_coord.x as i32, update_coord.y as i32],
                    flat_coord,
                    score,
                ));

                // If the list is getting sizeable, force flush it.
                if update_queue.len() >= FORCE_FLUSH_THRESHOLD {
                    // We drain the items out of the queue, and immediately unlock it for other threads.
                    // We can then proceed to flush the items from our local copy of the queue.
                    Some(update_queue.drain(..).collect())
                } else {
                    None
                }
            };

            if let Some(force_flush_items) = force_flush_items {
                self.flush_resolved(
                    &mut *self.rtree.write().unwrap(),
                    &force_flush_items,
                    is_tiling_mode,
                );
            } else {
                // Otherwise, check if we can get a lock on the rtree, and only then flush the list.
                // The rtree lock has moderate contention, so we might not get it this time around.
                if let Ok(ref mut rtree) = self.rtree.try_write() {
                    let update_queue: Vec<_> =
                        self.update_queue.lock().unwrap().drain(..).collect();

                    self.flush_resolved(&mut *rtree, &update_queue, is_tiling_mode);
                }
            }
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
        {
            let resolved = self.resolved.read().unwrap();

            //check how many resolved neighbors we have
            let total_resolved = resolved.len() as u32;
            if total_resolved == 0 {
                return false;
            } else if total_resolved <= k {
                //just return the resolved neighs we have
                k_neighs_2d.extend(
                    resolved
                        .iter()
                        .map(|(coord_flat, _)| coord_flat.to_2d(self.output_size).to_signed()),
                );
                return true;
            }
        }

        //return the search of the tree
        k_neighs_2d.extend(
            self.rtree
                .read()
                .unwrap()
                .nearest_neighbor_iter(&[coord.x as i32, coord.y as i32])
                .take(k as usize)
                .map(|a| SignedCoord2D::from((*a)[0], (*a)[1])),
        );
        true
    }

    fn get_distances_to_k_neighs(&self, coord: Coord2D, k_neighs_2d: &[SignedCoord2D]) -> Vec<f64> {
        let (dimx, dimy) = (f64::from(self.output_size.0), f64::from(self.output_size.1));
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
                    unresolved_flat.to_2d(self.output_size),
                    example_maps,
                    seed + i as u64 + u64::from(unresolved_flat.0),
                );
            }
        }
        self.locked_resolved += steps; //lock these pixels from being re-resolved
    }

    fn resolve_at_random(&self, my_coord: Coord2D, example_maps: &[ImageBuffer<'_>], seed: u64) {
        let rand_map: u32 = Pcg32::seed_from_u64(seed).gen_range(0, example_maps.len()) as u32;
        let rand_x: u32 =
            Pcg32::seed_from_u64(seed).gen_range(0, example_maps[rand_map as usize].width as u32);
        let rand_y: u32 =
            Pcg32::seed_from_u64(seed).gen_range(0, example_maps[rand_map as usize].height as u32);

        self.update(
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
        valid_samples_mask: &[SamplingMethod],
        m: u32,
        m_seed: u64,
    ) -> &'a [CandidateStruct] {
        let mut candidate_count = 0;
        let unresolved_coord = unresolved_coord.to_signed();

        let wrap_dim = (self.output_size.0 as i32, self.output_size.1 as i32);

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
            let (n_original_coord, _) = self.coord_map[n_flat_coord];
            let (n_patch_id, n_map_id) = self.id_map[n_flat_coord];
            //candidate coord is the original location of the neighbor + neighbor's shift to the center
            let candidate_coord = SignedCoord2D::from(
                n_original_coord.x as i32 + shift.0,
                n_original_coord.y as i32 + shift.1,
            );
            //check if the shifted coord is valid (discard if not)
            if check_coord_validity(
                candidate_coord,
                n_map_id,
                &example_maps,
                &valid_samples_mask[n_map_id.0 as usize],
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
        for _ in 0..m {
            let rand_map = (rng.gen_range(0, example_maps.len())) as u32;
            let dims = example_maps[rand_map as usize].dimensions();
            let mut rand_x: i32;
            let mut rand_y: i32;
            let mut candidate_coord;
            //generate a random valid candidate
            loop {
                rand_x = rng.gen_range(0, dims.0) as i32;
                rand_y = rng.gen_range(0, dims.1) as i32;
                candidate_coord = SignedCoord2D::from(rand_x, rand_y);
                if check_coord_validity(
                    candidate_coord,
                    MapId(rand_map),
                    &example_maps,
                    &valid_samples_mask[rand_map as usize],
                ) {
                    break;
                }
            }
            //for patch id (since we are not copying from a generated patch anymore), we take the pixel location in the example map
            let map_id = MapId(rand_map);
            let patch_id = PatchId(
                candidate_coord
                    .to_unsigned()
                    .to_flat((dims.0 as u32, dims.1 as u32))
                    .0,
            );
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
        let mut map_id_map = image::RgbaImage::new(self.output_size.0, self.output_size.1);
        let mut patch_id_map = image::RgbaImage::new(self.output_size.0, self.output_size.1);
        //populate the image with colors
        for (i, (patch_id, map_id)) in self.id_map.iter().enumerate() {
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
        let mut uncertainty_map = image::RgbaImage::new(self.output_size.0, self.output_size.1);

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

    //replace every resolved pixel with a pixel from a new level
    fn next_pyramid_level(&mut self, example_maps: &[ImageBuffer<'_>]) {
        for (coord_flat, _) in self.resolved.read().unwrap().iter() {
            let resolved_2d = coord_flat.to_2d(self.output_size);
            let (example_map_coord, example_map_id) = self.coord_map[coord_flat.0 as usize]; //so where the current pixel came from

            self.color_map[(resolved_2d.x, resolved_2d.y)] = *example_maps
                [example_map_id.0 as usize]
                .get_pixel(example_map_coord.x, example_map_coord.y);
        }
    }

    pub(crate) fn main_resolve_loop(
        &mut self,
        params: &GeneratorParams,
        example_maps_pyramid: &[ImagePyramid],
        mut progress: Option<Box<dyn crate::GeneratorProgress>>,
        guides_pyramid: &Option<GuidesPyramidStruct>,
        valid_samples: &[SamplingMethod],
    ) {
        let total_pixels_to_resolve = self.unresolved.lock().unwrap().len();
        let mut pyramid_level = 0;

        let stage_pixels_to_resolve = |p_stage: i32| {
            (params.p.powf(p_stage as f32) * (total_pixels_to_resolve as f32)) as usize
        };

        let actual_total_pixels_to_resolve =
            (0..=params.p_stages).map(stage_pixels_to_resolve).sum();

        let is_tiling_mode = params.tiling_mode;

        let cauchy_precomputed = PrerenderedU8Function::new(|a, b| {
            metric_cauchy(a, b, params.cauchy_dispersion * params.cauchy_dispersion)
        });
        let l2_precomputed = PrerenderedU8Function::new(metric_l2);
        let mut total_processed_pixels = 0;
        let max_workers = params.max_thread_count;

        for p_stage in (0..=params.p_stages).rev() {
            //get maps from current pyramid level (for now it will be p-stage dependant)
            let example_maps =
                get_single_example_level(&example_maps_pyramid, pyramid_level as usize);
            let guides = get_single_guide_level(&guides_pyramid, pyramid_level as usize);

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

            crossbeam_utils::thread::scope(|scope| {
                for _ in 0..n_workers {
                    scope.spawn(|_| {
                        let mut candidates: Vec<CandidateStruct> = Vec::new();
                        let mut candidates_patterns: Vec<ColorPattern> = Vec::new();
                        let mut my_pattern: ColorPattern = ColorPattern::new();
                        let mut k_neighs: Vec<SignedCoord2D> =
                            Vec::with_capacity(params.nearest_neighbors as usize);

                        let max_candidate_count = params.nearest_neighbors as usize
                            + params.random_sample_locations as usize;

                        candidates.resize(max_candidate_count, CandidateStruct::default());
                        candidates_patterns.resize(max_candidate_count, ColorPattern::new());

                        //alloc storage for our guides (regardless of whether we have them or not)
                        let mut my_guide_pattern: ColorPattern = ColorPattern::new();
                        let mut candidates_guide_patterns: Vec<ColorPattern> = Vec::new();
                        candidates_guide_patterns.resize(max_candidate_count, ColorPattern::new());

                        let out_color_map = &[ImageBuffer::from(&self.color_map)];

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
                                    &valid_samples,
                                    params.random_sample_locations as u32,
                                    loop_seed + 1,
                                );

                                // 3.1 get patterns for color maps
                                for (cand_i, cand) in candidates.iter().enumerate() {
                                    k_neighs_to_color_pattern(
                                        &cand.k_neighs,
                                        image::Rgba([0, 0, 0, 255]),
                                        &example_maps,
                                        &mut candidates_patterns[cand_i],
                                        false,
                                    );
                                }

                                let candidates_patterns = &candidates_patterns[0..candidates.len()];

                                k_neighs_to_color_pattern(
                                    &k_neighs_w_map_id, //feed into the function with always 0 index of the sample map
                                    image::Rgba([0, 0, 0, 255]),
                                    out_color_map,
                                    &mut my_pattern,
                                    is_tiling_mode,
                                );

                                // 3.2 get pattern for guide map if we have them
                                let (my_cost, guide_cost) = if let Some(ref in_guides) = guides {
                                    // populate guidance patterns for candidates
                                    for (cand_i, cand) in candidates.iter().enumerate() {
                                        k_neighs_to_color_pattern(
                                            &cand.k_neighs,
                                            image::Rgba([0, 0, 0, 255]),
                                            &in_guides.example_guides,
                                            &mut candidates_guide_patterns[cand_i],
                                            false,
                                        );

                                        //get example pattern to compare to
                                        k_neighs_to_color_pattern(
                                            &k_neighs_w_map_id,
                                            image::Rgba([0, 0, 0, 255]),
                                            &[in_guides.target_guide.clone()],
                                            &mut my_guide_pattern,
                                            is_tiling_mode,
                                        );
                                    }

                                    (
                                        &my_inverse_alpha_cost_precomputed,
                                        Some(&guide_cost_precomputed),
                                    )
                                } else {
                                    (&cauchy_precomputed, None)
                                };

                                let candidates_guide_patterns =
                                    &candidates_guide_patterns[0..candidates.len()];

                                // 4. find best match based on the candidate patterns
                                let (best_match, score) = find_best_match(
                                    &candidates,
                                    &my_pattern,
                                    &candidates_patterns,
                                    &my_guide_pattern,
                                    &candidates_guide_patterns,
                                    &k_neighs_dist,
                                    &my_cost,
                                    guide_cost,
                                );

                                let best_match_coord = best_match.coord.0.to_unsigned();
                                let best_match_map_id = best_match.coord.1;

                                // 5. resolve our pixel
                                self.update(
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
                                self.resolve_at_random(unresolved_2d, &example_maps, p_stage_seed);
                            }
                        }

                        remaining_threads.fetch_sub(1, Ordering::Relaxed);
                    });
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
                                image: &self.color_map,
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

            // Some items might still be pending a resolve flush. Do it now before we start the next stage.
            self.force_flush_resolved(is_tiling_mode);
        }
    }
}

fn k_neighs_to_color_pattern(
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
        let coord = if is_wrap_mode {
            n_coord.wrap(wrap_dim)
        } else {
            *n_coord
        };

        let end = i + 4;

        //check if he haven't gone outside the possible bounds
        if source_maps[n_map.0 as usize].is_in_bounds(coord) {
            pattern.0[i..end].copy_from_slice(
                &(source_maps[n_map.0 as usize])
                    .get_pixel(coord.x as u32, coord.y as u32)
                    .0[..4],
            )
        } else {
            // if we have gone out of bounds, then just fill as outside color
            pattern.0[i..end].copy_from_slice(&outside_color.0[..]);
        }

        i = end;
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

#[allow(clippy::too_many_arguments)]
fn find_best_match<'a>(
    candidates: &'a [CandidateStruct],
    my_pattern: &ColorPattern,
    candidates_patterns: &[ColorPattern],
    my_guide_pattern: &ColorPattern,
    candidates_guide_patterns: &[ColorPattern],
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

    for (i, (candidate_pattern, candidate_guide_pattern)) in candidates_patterns
        .iter()
        .zip(candidates_guide_patterns.iter())
        .enumerate()
    {
        if let Some(cost) = better_match(
            &my_pattern,
            candidate_pattern,
            &my_guide_pattern,
            candidate_guide_pattern,
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
    my_pattern: &ColorPattern,
    candidate_pattern: &ColorPattern,
    my_guide_pattern: &ColorPattern,
    candidate_guide_pattern: &ColorPattern,
    distance_gaussians: &[f32], //weight by distance
    my_cost: &PrerenderedU8Function,
    guide_cost: Option<&PrerenderedU8Function>,
    current_best: f32,
) -> Option<f32> {
    let mut score: f32 = 0.0; //minimize score

    for ((my_value, candidate_value), dist_gaussian) in my_pattern
        .0
        .iter()
        .copied()
        .zip(candidate_pattern.0.iter().copied())
        .zip(distance_gaussians.iter().copied())
    {
        score += dist_gaussian * my_cost.get(my_value, candidate_value);

        if score >= current_best {
            return None;
        }
    }

    if let Some(guide_cost_fn) = guide_cost {
        for ((my_guide, candidate_guide), dist_gaussian) in my_guide_pattern
            .0
            .iter()
            .copied()
            .zip(candidate_guide_pattern.0.iter().copied())
            .zip(distance_gaussians.iter().copied())
        {
            score += dist_gaussian * guide_cost_fn.get(my_guide, candidate_guide);

            if score >= current_best {
                return None;
            }
        }
    }

    Some(score)
}

struct PrerenderedU8Function {
    data: Vec<f32>,
}

impl PrerenderedU8Function {
    pub fn new<F: Fn(u8, u8) -> f32>(function: F) -> PrerenderedU8Function {
        let mut data = vec![0f32; 65536];

        for a in 0..=255u8 {
            for b in 0..=255u8 {
                data[a as usize * 256usize + b as usize] = function(a, b);
            }
        }

        PrerenderedU8Function { data }
    }

    #[inline]
    pub fn get(&self, a: u8, b: u8) -> f32 {
        self.data[a as usize * 256usize + b as usize]
    }
}

#[inline]
fn check_coord_validity(
    coord: SignedCoord2D,
    map_id: MapId,
    example_maps: &[ImageBuffer<'_>],
    mask: &SamplingMethod,
) -> bool {
    if mask.is_ignore() || !example_maps[map_id.0 as usize].is_in_bounds(coord) {
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
    pyramid_level: usize,
) -> Vec<ImageBuffer<'a>> {
    example_maps_pyramid
        .iter()
        .map(|a| ImageBuffer::from(&a.pyramid[pyramid_level]))
        .collect()
}

//get all the guide images from a single pyramid level
fn get_single_guide_level(
    guides_pyramid: &Option<GuidesPyramidStruct>,
    pyramid_level: usize,
) -> Option<GuidesStruct> {
    if let Some(ref guides_pyr) = guides_pyramid {
        Some(guides_pyr.to_guides_struct(pyramid_level))
    } else {
        None
    }
}
