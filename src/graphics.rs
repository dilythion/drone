use std::{sync::Arc, time::Duration};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::VulkanoWindows,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};
use rand::Rng;

enum State {
    Playground,
    Guessing {
        answer: [u8; 2],
    },
    Guessed {
        answer: [u8; 2],
    },
}

enum Command {
    Toggle,
    On,
    Off,
}

struct RenderContext {
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
}

pub struct App<'a> {
    context: VulkanoContext,
    windows: VulkanoWindows,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[Vertex2D]>,
    index_buffer: Subbuffer<[u16]>,
    rcx: Option<RenderContext>,
    selected: [u8; 2],
    position: [f32; 2],
    directions: [Direction; 4],
    activated: u64,
    motion_count: Option<u8>,
    motion_command: Option<Command>,
    last_frame: std::time::Instant,
    tx: std::sync::mpsc::Sender<crate::Request>,
    rng: rand::rngs::ThreadRng,
    drone: f32,
    state: State,
    config: &'a crate::config::Config,
}

impl<'a> App<'a> {
    pub fn new(_event_loop: &EventLoop<()>, tx: std::sync::mpsc::Sender<crate::Request>, config: &'a crate::config::Config) -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());

        let windows = VulkanoWindows::default();

        println!(
            "Using device: {} (type: {:?})",
            context.device().physical_device().properties().device_name,
            context.device().physical_device().properties().device_type,
        );

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        let vertex_buffer = Buffer::from_iter(
            context.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [
                Vertex2D { position: [0.0, 0.0], },
                Vertex2D { position: [1.0, 0.0], },
                Vertex2D { position: [0.0, 1.0], },
                Vertex2D { position: [1.0, 1.0], },
            ]
        )
        .unwrap();
        let index_buffer = Buffer::from_iter(
            context.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [
                0, 1, 2,
                1, 2, 3,
            ],
        )
        .unwrap();

        App {
            context,
            windows,
            command_buffer_allocator,
            vertex_buffer,
            index_buffer,
            rcx: None,
            selected: [0; 2],
            position: [0.0; 2],
            directions: [Direction::default(); 4],
            activated: 0b000000000000_111111111111_111111111111_000000000000,
            motion_count: None,
            motion_command: None,
            last_frame: std::time::Instant::now(),
            tx,
            rng: rand::rng(),
            drone: 0.0,
            state: State::Playground,
            config,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Direction {
    active: bool,
    last: Option<std::time::SystemTime>,
    first: bool,
}

impl Direction {
    fn iteration(&mut self, interval: u128, motion_count: &mut Option<u8>, state: &mut State) -> u8 {
        if self.first { 
            self.first = false;
            let count = motion_count.unwrap_or(1);
            *motion_count = None;
            if let State::Guessed { .. } = state { *state = State::Playground; }
            return count;
        }
        if let Some(last) = self.last {
            if let Ok(elapsed) = last.elapsed() {
                if elapsed.as_millis() >= interval {
                    self.last = Some(std::time::SystemTime::now());
                    return 1;
                }
            }        
        }
        0
    }

    fn on(&mut self) {
        self.active = true;
        self.last = Some(
            std::time::SystemTime::now()
                .checked_add(std::time::Duration::from_millis(500))
                .unwrap()
        );
        self.first = true;
    }

    fn off(&mut self) {
        self.active = false;
        self.last = None;
        self.first = false;
    }
}

fn tritone_equivalent(x: u8, y: u8) -> u8 {
    y * 12 + if x < 12 { x } else { 0 }
}

fn random_drone(rng: &mut rand::rngs::ThreadRng, drone: &mut f32) -> f32 {
    let number = loop {
        let number = rng.random_range(0..12) as f32;
        if (number - *drone).abs() > 0.1 { break number; }
    }; 
    *drone = number;
    440.0 * 2.0_f32.powf(number / 12.0)
}

fn position_note(mut x: u8, y: u8, drone: f32) -> f32 {
    x = if x < 12 { x } else { 0 };
    x = [6, 1, 8, 3, 10, 5, 0, 7, 2, 9, 4, 11][x as usize];
    if x == 0 && y < 2 { x = 12; }
    let number: i8 = x as i8 + (1 - y as i8) * 12; 
    let number: f32 = number as f32 + drone;
    440.0 * 2.0_f32.powf(number / 12.0)
}

fn random_active_note(activated: u64, rng: &mut rand::rngs::ThreadRng, last: Option<[u8; 2]>) -> [u8; 2] {
    let option_count: u8 = activated.count_ones() as u8;
    assert!(option_count > 2);
    loop {
        let chosen: u8 = rng.random_range(0..option_count);
        let mut output = [0, 0];
        let mut i: u8 = 0;
        let mut activated_copy = activated;
        loop {
            if activated_copy & 1 == 1 { i += 1; }
            if i == chosen + 1 {
                if let Some(last) = last {
                    if output == last { break; }
                }
                return output;
            }
            activated_copy >>= 1;
            output[0] += 1;
            if output[0] == 12 {
                output[0] = 0;
                output[1] += 1;
            }
        }
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(primary_window_id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(primary_window_id);
        }

        let id = self.windows
            .create_window(event_loop, &self.context, &Default::default(), |_| {});
        self.windows.get_window(id).unwrap().set_cursor_visible(false);
        self.windows.get_window(id).unwrap().set_title("Relative Pitch Is The Most Important Skill In Music");
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let window_size = window_renderer.window().inner_size();

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec2 position;
                    layout(location = 1) in vec3 color;
                    layout(location = 2) in vec2 corner;
                    layout(location = 3) in vec2 size;

                    layout(location = 0) out vec3 color_out;
                    
                    void main() {
                        gl_Position = vec4(position * size + corner, 0.0, 1.0);
                        color_out = color;
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 color_out;

                    layout(location = 0) out vec4 f_color;

                    void main() {
                        f_color = vec4(color_out, 1.0);
                    }
                ",
            }
        }

        let render_pass = vulkano::single_pass_renderpass!(
            self.context.device().clone(),
            attachments: {
                color: {
                    format: window_renderer.swapchain_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let framebuffers =
            window_size_dependent_setup(window_renderer.swapchain_image_views(), &render_pass);

        let pipeline = {
            let vs = vs::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = [Vertex2D::per_vertex(), Quad::per_instance()].definition(&vs).unwrap();
            
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.context.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.context.device().clone())
                    .unwrap(),
            ).unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                self.context.device().clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        self.rcx = Some(RenderContext {
            render_pass,
            framebuffers,
            pipeline,
            viewport,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                window_renderer.resize();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                type Key = winit::keyboard::PhysicalKey;
                type Code = winit::keyboard::KeyCode;
                match event.state {
                    winit::event::ElementState::Pressed => {
                        match event.physical_key {
                            Key::Code(Code::KeyD) => {
                                if let State::Guessing { .. } = self.state { return; }
                                if let State::Guessed { .. } = self.state { self.state = State::Playground; } 
                                self.tx.send(crate::Request::SetDrone(random_drone(&mut self.rng, &mut self.drone))).unwrap();
                            }
                            Key::Code(Code::KeyF) => {
                                if self.activated & (1 << tritone_equivalent(self.selected[0], self.selected[1])) == 0 { return; }
                                self.tx.send(crate::Request::AddNote(position_note(self.selected[0], self.selected[1], self.drone))).unwrap(); 
                                if let State::Guessing { answer } = self.state {
                                    self.state = State::Guessed { answer, };
                                }
                            }
                            Key::Code(Code::KeyA) => {
                                match self.state {
                                    State::Playground => {
                                        if self.activated.count_ones() < 3 { return; }
                                        let answer = random_active_note(self.activated, &mut self.rng, None);
                                        self.state = State::Guessing { answer, }; 
                                        self.tx.send(crate::Request::AddNote(position_note(answer[0], answer[1], self.drone))).unwrap();
                                    }
                                    State::Guessed { answer, .. } => {
                                        if self.activated.count_ones() < 3 { return; }
                                        let answer = random_active_note(self.activated, &mut self.rng, Some(answer));
                                        self.state = State::Guessing { answer, }; 
                                        self.tx.send(crate::Request::AddNote(position_note(answer[0], answer[1], self.drone))).unwrap();
                                    }
                                    State::Guessing { answer } => 
                                        self.tx.send(crate::Request::AddNote(position_note(answer[0], answer[1], self.drone))).unwrap(),
                                }
                            }
                            Key::Code(Code::KeyH) => if !(self.directions[0].active) { self.directions[0].on(); },
                            Key::Code(Code::KeyL) => if !(self.directions[1].active) { self.directions[1].on(); },
                            Key::Code(Code::KeyK) => if !(self.directions[2].active) { self.directions[2].on(); },
                            Key::Code(Code::KeyJ) => if !(self.directions[3].active) { self.directions[3].on(); },
                            Key::Code(Code::KeyS) => {
                                if let State::Guessing { .. } = self.state { return; } 
                                if let State::Guessed { .. } = self.state { self.state = State::Playground; } 
                                if let Some(count) = self.motion_count {
                                    if count > 1 {
                                        self.motion_command = Some(Command::Toggle);
                                        return;
                                    }
                                }
                                self.activated ^= 1 << tritone_equivalent(self.selected[0], self.selected[1]);
                            }
                            Key::Code(Code::KeyW) => {
                                if let State::Guessing { .. } = self.state { return; } 
                                if let State::Guessed { .. } = self.state { self.state = State::Playground; } 
                                if let Some(count) = self.motion_count {
                                    if count > 1 {
                                        self.motion_command = Some(Command::On);
                                        return;
                                    }
                                }
                                self.activated |= 1 << tritone_equivalent(self.selected[0], self.selected[1]);
                            }
                            Key::Code(Code::KeyE) => {
                                if let State::Guessing { .. } = self.state { return; } 
                                if let State::Guessed { .. } = self.state { self.state = State::Playground; } 
                                if let Some(count) = self.motion_count {
                                    if count > 1 {
                                        self.motion_command = Some(Command::Off);
                                        return;
                                    }
                                }
                                self.activated &= !(1 << tritone_equivalent(self.selected[0], self.selected[1]));
                            }
                            Key::Code(Code::Digit1) => self.motion_count = Some(1),
                            Key::Code(Code::Digit2) => self.motion_count = Some(2),
                            Key::Code(Code::Digit3) => self.motion_count = Some(3),
                            Key::Code(Code::Digit4) => self.motion_count = Some(4),
                            Key::Code(Code::Digit5) => self.motion_count = Some(5),
                            Key::Code(Code::Digit6) => self.motion_count = Some(6),
                            Key::Code(Code::Digit7) => self.motion_count = Some(7),
                            Key::Code(Code::Digit8) => self.motion_count = Some(8),
                            Key::Code(Code::Digit9) => self.motion_count = Some(9),
                            _ => {}
                        }
                    }
                    winit::event::ElementState::Released => {
                        match event.physical_key {
                            Key::Code(Code::KeyH) => self.directions[0].off(),
                            Key::Code(Code::KeyL) => self.directions[1].off(),
                            Key::Code(Code::KeyK) => self.directions[2].off(),
                            Key::Code(Code::KeyJ) => self.directions[3].off(),
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let window_size = window_renderer.window().inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                let previous_frame_end = window_renderer
                    .acquire(Some(Duration::from_millis(1000)), |swapchain_images| {
                        rcx.framebuffers =
                            window_size_dependent_setup(swapchain_images, &rcx.render_pass);
                        rcx.viewport.extent = window_size.into();
                    })
                    .unwrap();

                fn apply_command(used: &mut bool, command: &Option<Command>, selected: &[u8; 2], activated: &mut u64) {
                    *used = true;
                    match command {
                        Some(Command::Toggle) => 
                            *activated ^= 1 << tritone_equivalent(selected[0], selected[1]),
                        Some(Command::On) => 
                            *activated |= 1 << tritone_equivalent(selected[0], selected[1]),
                        Some(Command::Off) => 
                            *activated &= !(1 << tritone_equivalent(selected[0], selected[1])),
                        _ => {},
                    }
                }

                let mut used = false;
                for _ in 0..self.directions[0].iteration(100, &mut self.motion_count, &mut self.state) {
                    apply_command(&mut used, &self.motion_command, &self.selected, &mut self.activated);
                    if self.selected[0] > 0 { self.selected[0] -= 1; }
                    else { self.selected[0] = 11; }
                }
                for _ in 0..self.directions[1].iteration(100, &mut self.motion_count, &mut self.state) {
                    apply_command(&mut used, &self.motion_command, &self.selected, &mut self.activated);
                    if self.selected[0] < 12 { self.selected[0] += 1; }
                    else { self.selected[0] = 1; }
                }
                for _ in 0..self.directions[2].iteration(200, &mut self.motion_count, &mut self.state) {
                    apply_command(&mut used, &self.motion_command, &self.selected, &mut self.activated);
                    if self.selected[1] > 0 { self.selected[1] -= 1; }
                    else { self.selected[1] = 3; }
                }
                for _ in 0..self.directions[3].iteration(200, &mut self.motion_count, &mut self.state) {
                    apply_command(&mut used, &self.motion_command, &self.selected, &mut self.activated);
                    if self.selected[1] < 3 { self.selected[1] += 1; }
                    else { self.selected[1] = 0; }
                }
                if used { self.motion_command = None; }

                fn brightness(x: i32, y: i32) -> f32 {
                    x as f32 * 0.05 - y as f32 * 0.1
                }

                fn multiply_brightness(color: [f32; 3], multiplier: f32) -> [f32; 3] {
                    [color[0] * multiplier, color[1] * multiplier, color[2] * multiplier]
                }

                let base_square_size = 0.05;
                let square_size = if window_size.height < window_size.width {
                    [base_square_size * window_size.height as f32 / window_size.width as f32, base_square_size]
                } else {
                    [base_square_size, base_square_size * window_size.width as f32 / window_size.height as f32]
                };

                let mut quads = vec![];

                let mut position = 0.0;
                for x in 0..13 {
                    let width = match x {
                        6 => 1.5,
                        0 | 12 => 0.5,
                        _ => 1.0,
                    };
                    let mut total_height = 0.0;
                    for y in 0..4 {
                        let height = match y {
                            1 | 2 => 1.1,
                            _ => 0.9,
                        };
                        let color = multiply_brightness(match (x + 2) % 3 {
                            0 => [0.4 + brightness(x, y), 0.0, 0.0],
                            1 => [0.4 + brightness(x, y), 0.4 + brightness(x, y), 0.0],
                            _ => [0.0, 0.4 + brightness(x, y), 0.0],
                        }, 1.0);
                        let new_quad = Quad {
                            color: if self.activated & (1 << tritone_equivalent(x as u8, y as u8)) == 0 { 
                                multiply_brightness(color, 0.05) 
                            } else { 
                                match self.state {
                                    State::Guessed { answer, .. } => {
                                        if [x as u8, y as u8] == answer 
                                            || (answer[0] == 0 && x == 12 && y as u8 == answer[1]) {
                                            color 
                                        } else {
                                            multiply_brightness(color, 0.1) 
                                        }
                                    }
                                    _ => color,
                                }
                            },
                            corner: [(position / 12.5) * 2.0 - 1.0, (total_height / 4.0) * 2.0 - 1.0],
                            size: [width * 2.0 / 12.5, height * 2.0 / 4.0],
                        };
                        quads.push(new_quad.clone());
                        if x == self.selected[0] as i32 && y == self.selected[1] as i32 {
                            let target = [
                                new_quad.corner[0] + new_quad.size[0] / 2.0,
                                new_quad.corner[1] + new_quad.size[1] / 2.0,
                            ];
                            let multiplier = self.config.animation.selector.unwrap_or(0.1) 
                                * (self.last_frame.elapsed().as_nanos() as f32 / (1_000_000_000 / 240) as f32);
                            self.last_frame = std::time::Instant::now();
                            self.position[0] += (target[0] - self.position[0]) * multiplier;
                            self.position[1] += (target[1] - self.position[1]) * multiplier;
                        }
                        total_height += height;
                    }
                    position += width;
                }
                let size = match self.state {
                    State::Guessing { .. } => [square_size[0] * 1.2, square_size[1] * 1.2],
                    _ => square_size,
                };
                quads.push(Quad {
                    color: [0.0, 0.0, 0.0],
                    corner: self.position,
                    size,
                }.center());
                let quad_input_buffer = Buffer::from_iter(
                    self.context.memory_allocator().clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    quads,
                )
                .unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.context.graphics_queue().queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[window_renderer.image_index() as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect()).unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone()).unwrap()
                    .bind_vertex_buffers(0, (self.vertex_buffer.clone(), quad_input_buffer.clone())).unwrap()
                    .bind_index_buffer(self.index_buffer.clone()).unwrap();

                unsafe { builder.draw_indexed(self.index_buffer.len() as u32, quad_input_buffer.len() as u32, 0, 0, 0) }.unwrap();

                builder
                    .end_render_pass(Default::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .then_execute(self.context.graphics_queue().clone(), command_buffer)
                    .unwrap()
                    .boxed();

                window_renderer.present(future, false);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        window_renderer.window().request_redraw();
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
struct Quad {
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
    #[format(R32G32_SFLOAT)]
    corner: [f32; 2],
    #[format(R32G32_SFLOAT)]
    size: [f32; 2],
}

impl Quad {
    fn center(&self) -> Self {
        Self {
            corner: [
                self.corner[0] - self.size[0] / 2.0,
                self.corner[1] - self.size[1] / 2.0,
            ],
            ..*self
        }
    }
}

fn window_size_dependent_setup(
    swapchain_images: &[Arc<ImageView>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    swapchain_images
        .iter()
        .map(|swapchain_image| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![swapchain_image.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
