use winit::event_loop::EventLoop;
use cpal::traits::StreamTrait;

mod audio;
mod graphics;   
mod config;

fn main() -> anyhow::Result<()> {
    let config = config::Config::new();

    let (tx, rx) = std::sync::mpsc::channel::<Request>();

    let stream = audio::stream_setup_for(rx, config.clone())?;
    stream.play()?;

    let event_loop = EventLoop::new().unwrap();
    let mut app = graphics::App::new(&event_loop, tx, &config);

    Ok(event_loop.run_app(&mut app)?)
}

enum Request {
    AddNote(f32),
    SetDrone(f32),
}

