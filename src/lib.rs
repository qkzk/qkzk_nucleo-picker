//! # A generic fuzzy item picker
//! This is a generic picker implementation which wraps the [`nucleo::Nucleo`] matching engine with
//! an interactive TUI.
//!
//! The API is pretty similar to that exposed by the [`nucleo`] crate; the majority of the internal state of [`Nucleo`] is re-exposed through the main [`Picker`] entrypoint.
//!
//! For usage examples, visit the [examples
//! folder](https://github.com/autobib/nucleo-picker/tree/master/examples) on GitHub.
mod bind;
pub mod component;
pub mod fill;

use std::{
    cmp::min,
    io::{self, Error, ErrorKind},
    process::{Child, Command, Stdio},
    sync::Arc,
    thread::{available_parallelism, sleep},
    time::{Duration, Instant},
};

use crate::{
    bind::{convert, Event},
    component::{Edit, EditableString},
};
use anyhow::{bail, Result};
use component::View;
use crossterm::{
    event::{poll, read},
    terminal::size,
    tty::IsTty,
};
use nucleo::{Config, Injector, Nucleo, Utf32String};
use ratatui::{
    layout::{Constraint, Direction, Layout, Position, Rect},
    restore,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    DefaultTerminal, Frame,
};

pub use nucleo;

/// The outcome after processing all of the events.
enum EventSummary {
    /// Continue rendering the frame.
    Continue,
    /// The prompt was updated; where the updates append-only?
    UpdatePrompt(bool),
    /// Select the given item and quit.
    Select,
    /// Quit without selecting an item.
    Quit,
}

/// The dimension parameters of various items in the screen.
#[derive(Debug)]
struct Dimensions {
    /// The width of the screen.
    width: u16,
    /// The height of the screen, including the prompt.
    height: u16,
    /// The left buffer size of the prompt.
    prompt_left_padding: u16,
    /// The right buffer size of the prompt.
    prompt_right_padding: u16,
}

impl Dimensions {
    /// Initialize based on screen dimensions.
    pub fn from_screen(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            prompt_left_padding: width / 8,
            prompt_right_padding: width / 12,
        }
    }

    /// The maximum width of the prompt string display window.
    pub fn prompt_max_width(&self) -> usize {
        self.width
            .saturating_sub(self.prompt_left_padding)
            .saturating_sub(self.prompt_right_padding)
            .saturating_sub(2) as _
    }

    /// The maximum number of matches which can be drawn to the screen.
    pub fn max_draw_count(&self) -> u16 {
        self.height.saturating_sub(2)
    }

    /// The maximum length on which a match can be drawn.
    pub fn max_draw_length(&self) -> u16 {
        self.width.saturating_sub(2)
    }
}

/// A representation of the current state of the picker.
#[derive(Debug)]
struct PickerState {
    /// The dimensions of the application.
    dimensions: Dimensions,
    /// The selector index position, or [`None`] if there is nothing to select.
    selector_index: Option<u16>,
    /// The prompt string.
    prompt: EditableString,
    /// The current number of items to be drawn to the terminal.
    draw_count: u16,
    /// The total number of items.
    item_count: u32,
    /// The number of matches.
    matched_item_count: u32,
    /// Has the state changed?
    needs_redraw: bool,
    /// Previewer
    previewer: Option<Previewer>,
}

impl PickerState {
    /// The initial picker state.
    pub fn new(screen: (u16, u16), previewer: Option<Previewer>) -> Self {
        let dimensions = Dimensions::from_screen(screen.0, screen.1);
        let prompt = EditableString::new(dimensions.prompt_max_width());

        Self {
            dimensions,
            selector_index: None,
            prompt,
            draw_count: 0,
            matched_item_count: 0,
            item_count: 0,
            needs_redraw: true,
            previewer,
        }
    }

    /// Increment the current item selection.
    pub fn incr_selection(&mut self) {
        self.needs_redraw = true;
        self.selector_index = self.selector_index.map(|i| i.saturating_add(1));
        self.clamp_selector_index();
    }

    /// Decrement the current item selection.
    pub fn decr_selection(&mut self) {
        self.needs_redraw = true;
        self.selector_index = self.selector_index.map(|i| i.saturating_sub(1));
        self.clamp_selector_index();
    }

    /// Update the draw count from a snapshot.
    pub fn update<T: Send + Sync + 'static>(
        &mut self,
        changed: bool,
        snapshot: &nucleo::Snapshot<T>,
    ) {
        if changed {
            self.needs_redraw = true;
            self.item_count = snapshot.item_count();
            self.matched_item_count = snapshot.matched_item_count();
            self.draw_count = self.matched_item_count.try_into().unwrap_or(u16::MAX);
            self.clamp_draw_count();
            self.clamp_selector_index();
        }
    }

    /// Clamp the draw count so that it falls in the valid range.
    fn clamp_draw_count(&mut self) {
        self.draw_count = min(self.draw_count, self.dimensions.max_draw_count())
    }

    /// Clamp the selector index so that it falls in the valid range.
    fn clamp_selector_index(&mut self) {
        if self.draw_count == 0 {
            self.selector_index = None;
        } else {
            let position = min(self.selector_index.unwrap_or(0), self.draw_count - 1);
            self.selector_index = Some(position);
        }
    }

    /// Perform the given edit action.
    pub fn edit_prompt(&mut self, st: Edit) {
        self.needs_redraw |= self.prompt.edit(st);
    }

    /// Clear the queued events.
    fn handle(&mut self) -> Result<EventSummary, io::Error> {
        let mut update_prompt = false;
        let mut append = true;

        while poll(Duration::from_millis(5))? {
            if let Some(event) = convert(read()?) {
                match event {
                    Event::MoveToStart => self.edit_prompt(Edit::ToStart),
                    Event::MoveToEnd => self.edit_prompt(Edit::ToEnd),
                    Event::Insert(ch) => {
                        update_prompt = true;
                        // if the cursor is at the end, it means the character was appended
                        append &= self.prompt.cursor_at_end();
                        self.edit_prompt(Edit::Insert(ch));
                    }
                    Event::Select => return Ok(EventSummary::Select),
                    Event::MoveUp => self.decr_selection(),
                    Event::MoveDown => self.incr_selection(),
                    Event::MoveLeft => self.edit_prompt(Edit::Left),
                    Event::MoveRight => self.edit_prompt(Edit::Right),
                    Event::Delete => {
                        update_prompt = true;
                        append = false;
                        self.edit_prompt(Edit::Delete);
                    }
                    Event::Quit => return Ok(EventSummary::Quit),
                    Event::Resize(width, height) => {
                        self.resize(width, height);
                    }
                    Event::Paste(contents) => {
                        update_prompt = true;
                        append &= self.prompt.cursor_at_end();
                        self.edit_prompt(Edit::Paste(contents));
                    }
                }
            }
        }
        Ok(if update_prompt {
            EventSummary::UpdatePrompt(append)
        } else {
            EventSummary::Continue
        })
    }

    /// Resize the terminal state on screen size change.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.needs_redraw = true;
        self.dimensions = Dimensions::from_screen(width, height);
        self.prompt.resize(self.dimensions.prompt_max_width());
        self.clamp_draw_count();
        self.clamp_selector_index();
    }
}

struct Display<'a, T: Send + Sync + 'static> {
    picker_state: &'a PickerState,
    snapshot: &'a nucleo::Snapshot<T>,
}

impl<'a, T: Send + Sync + 'static> Display<'a, T> {
    fn new(picker_state: &'a PickerState, snapshot: &'a nucleo::Snapshot<T>) -> Self {
        Self {
            picker_state,
            snapshot,
        }
    }

    fn draw(&mut self, term: &'a mut DefaultTerminal) -> Result<()> {
        term.draw(|f| {
            let rects = self.build_layout(f.area());

            // Draw the match counts at the top
            let match_info = self.line_match_info();
            let match_count_paragraph = Self::paragraph_match_count(match_info);
            f.render_widget(match_count_paragraph, rects[0]);

            // Draw the matched items
            let (items_paragraph, previewed) = self.paragraph_matches();
            f.render_widget(items_paragraph, rects[1]);
            if previewed.is_some() {
                self.draw_preview(f, rects[3], previewed);
            }

            // Draw the prompt at the bottom
            self.draw_prompt(f, rects[2]);
        })?;
        Ok(())
    }

    fn build_layout(&self, area: Rect) -> Vec<Rect> {
        if self.picker_state.previewer.is_none() {
            Self::rect_without_preview(area)
        } else {
            Self::rects_with_preview(area)
        }
    }

    fn rect_without_preview(area: Rect) -> Vec<Rect> {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(0),
                Constraint::Min(0),
            ])
            .split(area)
            .to_vec()
    }

    fn rects_with_preview(area: Rect) -> Vec<Rect> {
        let horizontal = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        let mut chunks = Self::rect_without_preview(horizontal[0]);
        chunks.push(horizontal[1]);
        chunks
    }

    fn line_match_info(&self) -> Line {
        Line::from(vec![
            Span::styled("  ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{}", self.picker_state.matched_item_count),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::ITALIC),
            ),
            Span::styled("/", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{}", self.picker_state.item_count),
                Style::default().fg(Color::Yellow),
            ),
        ])
    }

    fn paragraph_match_count(match_info: Line) -> Paragraph {
        Paragraph::new(match_info)
            .style(Style::default())
            .block(Block::default().borders(Borders::NONE))
    }

    fn paragraph_matches(&self) -> (Paragraph, Option<String>) {
        let mut items: Vec<Line> = vec![];
        let mut previewed = None;

        for (index, item) in self
            .snapshot
            .matched_items(..self.picker_state.draw_count as u32)
            .enumerate()
        {
            let render = self.format_utf32(&item.matcher_columns[0]);

            let item_spans = if Some(index) == self.picker_state.selector_index.map(|i| i as _) {
                if self.picker_state.previewer.is_some() {
                    previewed = Some(render.clone());
                }
                Self::selected_line(render)
            } else {
                Self::non_selected_line(render)
            };
            items.push(item_spans);
        }
        (Paragraph::new(items), previewed)
    }

    /// Format a [`Utf32String`] for displaying. Currently:
    /// - Delete control characters.
    /// - Truncates the string to an appropriate length.
    /// - Replaces any newline characters with spaces.
    fn format_utf32(&self, utf32: &Utf32String) -> String {
        utf32
            .slice(..)
            .chars()
            .filter(|ch| !ch.is_control())
            .take(self.picker_state.dimensions.max_draw_length() as _)
            .map(|ch| match ch {
                '\n' => ' ',
                s => s,
            })
            .collect()
    }

    fn draw_preview(&self, f: &mut Frame, rect: Rect, previewed: Option<String>) {
        let Some(previewer) = &self.picker_state.previewer else {
            return;
        };
        let Some(previewed) = previewed else {
            return;
        };
        f.render_widget(
            Self::paragraph_preview(&previewer.run(&previewed).unwrap()),
            rect,
        );
    }

    fn paragraph_preview(preview_output: &str) -> Paragraph {
        Paragraph::new(preview_output).block(Block::default().borders(Borders::LEFT))
    }

    fn selected_line(render: String) -> Line<'static> {
        Line::from(vec![
            Span::styled(
                "▌ ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(render),
        ])
    }

    fn non_selected_line(render: String) -> Line<'static> {
        Line::from(vec![Span::raw("  "), Span::raw(render)])
    }

    fn draw_prompt(&self, f: &mut Frame, rect: Rect) {
        // Render the prompt string at the bottom
        let prompt_view = self.prompt_view();
        let prompt_paragraph = Paragraph::new(vec![Line::from(vec![
            Span::raw("> "),
            Span::raw(prompt_view.to_string()),
        ])])
        .block(Block::default().borders(Borders::NONE));

        // Render the prompt at the bottom of the layout
        f.render_widget(prompt_paragraph, rect);
        self.set_cursor_position(f, rect, &prompt_view);
    }

    fn prompt_view(&self) -> View<char> {
        self.picker_state.prompt.view_padded(
            self.picker_state.dimensions.prompt_left_padding as _,
            self.picker_state.dimensions.prompt_right_padding as _,
        )
    }

    fn set_cursor_position(&self, f: &mut Frame, rect: Rect, prompt_view: &View<char>) {
        // Move the cursor to the prompt
        f.set_cursor_position(Position {
            x: rect.x + prompt_view.index() as u16 + 2, // Adjust the cursor position for "> "
            y: rect.y,
        });
    }
}

/// # The core item picker
/// This is the main entrypoint for this crate. Initialize a picker with the [`Picker::new`] or the
/// [`Picker::default`] implementation and add elements to the picker using an [`Injector`]
/// returned by the [`Picker::injector`] method.
///
/// See also the documentation for [`nucleo::Nucleo`] and [`nucleo::Injector`], or the
/// [usage examples](https://github.com/autobib/nucleo-picker/tree/master/examples).
pub struct Picker<T: Send + Sync + 'static> {
    matcher: Nucleo<T>,
    must_reset_term: bool,
    previewer: Option<Previewer>,
    term: DefaultTerminal,
}

impl<T: Send + Sync + 'static> Default for Picker<T> {
    fn default() -> Self {
        Self::new(Config::DEFAULT, Self::default_thread_count(), 1)
    }
}

impl<T: Send + Sync + 'static> Picker<T> {
    /// Best-effort guess to reduce thread contention. Reserve two threads:
    /// 1. for populating the macher
    /// 2. for rendering the terminal UI and handling user input
    fn default_thread_count() -> Option<usize> {
        available_parallelism()
            .map(|it| it.get().checked_sub(2).unwrap_or(1))
            .ok()
    }

    /// Default frame interval of 16ms, or ~60 FPS.
    const fn default_frame_interval() -> Duration {
        Duration::from_millis(16)
    }

    /// Create a new [`Picker`] instance with arguments passed to [`Nucleo`].
    pub fn new(config: Config, num_threads: Option<usize>, columns: u32) -> Self {
        Self {
            matcher: Nucleo::new(config, Arc::new(|| {}), num_threads, columns),
            must_reset_term: true,
            previewer: None,
            term: ratatui::init(),
        }
    }

    /// Create a new [`Picker`] instance with the given configuration.
    pub fn with_config(config: Config) -> Self {
        Self::new(config, Self::default_thread_count(), 1)
    }

    pub fn without_reset(mut self) -> Self {
        self.must_reset_term = false;
        self
    }

    pub fn with_previewer(mut self, command: String) -> Self {
        self.previewer = Some(Previewer::new(command));
        self
    }

    /// Get an [`Injector`] from the internal [`Nucleo`] instance.
    pub fn injector(&self) -> Injector<T> {
        self.matcher.injector()
    }

    /// Open the interactive picker prompt and return the picked item, if any.
    pub fn pick(&mut self) -> Result<Option<&T>> {
        eprintln!("enabled raw mode");
        if !std::io::stdin().is_tty() {
            bail!("is not interactive");
        }
        self.pick_inner(Self::default_frame_interval())
    }

    /// The actual picker implementation.
    fn pick_inner(&mut self, interval: Duration) -> Result<Option<&T>> {
        let mut picker_state = PickerState::new(size()?, self.previewer.clone());

        let selection = loop {
            let deadline = Instant::now() + interval;

            // process any queued keyboard events and reset pattern if necessary
            match picker_state.handle()? {
                EventSummary::Continue => {}
                EventSummary::UpdatePrompt(append) => {
                    self.matcher.pattern.reparse(
                        0,
                        &picker_state.prompt.full_contents(),
                        nucleo::pattern::CaseMatching::Smart,
                        nucleo::pattern::Normalization::Smart,
                        append,
                    );
                }
                EventSummary::Select => {
                    break picker_state
                        .selector_index
                        .and_then(|idx| self.matcher.snapshot().get_matched_item(idx as _))
                        .map(|it| it.data);
                }
                EventSummary::Quit => {
                    break None;
                }
            };

            // increment the matcher and update state
            let status = self.matcher.tick(10);
            picker_state.update(status.changed, self.matcher.snapshot());

            // redraw the screen
            if picker_state.needs_redraw {
                Display::new(&picker_state, self.matcher.snapshot()).draw(&mut self.term)?;
            }

            // wait if frame rendering finishes early
            sleep(deadline - Instant::now());
        };

        if self.must_reset_term {
            restore();
            eprintln!("disabled raw mode");
        }
        Ok(selection)
    }
}

#[derive(Default, Clone, Debug)]
struct Previewer {
    command: String,
}

impl Previewer {
    fn new(command: String) -> Self {
        Self { command }
    }

    fn run(&self, content: &str) -> Result<String> {
        let args = self.parse(content);
        let child = Self::spawn(args)?;
        Self::execute_and_output(child)
    }

    fn parse(&self, content: &str) -> Vec<String> {
        self.command
            .split(' ')
            .map(|arg| match arg {
                "%t" => content.to_owned(),
                arg => arg.to_owned(),
            })
            .collect()
    }

    fn spawn(mut args: Vec<String>) -> io::Result<Child> {
        if args.is_empty() {
            return Err(Error::new(ErrorKind::InvalidData, "Empty preview command"));
        }
        let program = args.remove(0);
        Command::new(program)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
    }

    fn execute_and_output(child: Child) -> Result<String> {
        let output = child.wait_with_output()?;
        let result = if output.status.success() {
            output.stdout
        } else {
            output.stderr
        };
        Ok(String::from_utf8(result)?)
    }
}
