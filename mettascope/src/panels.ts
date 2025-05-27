import { Vec2f, Mat3f } from './vector_math.js';
import * as Common from './common.js';
import { ui } from './common.js';
import { find } from './htmlutils.js'

export class PanelInfo {
  public x: number = 0;
  public y: number = 0;
  public width: number = 0;
  public height: number = 0;
  public name: string = "";
  public isPanning: boolean = false;
  public panPos: Vec2f = new Vec2f(0, 0);
  public zoomLevel: number = Common.DEFAULT_ZOOM_LEVEL;
  public div: HTMLElement;

  constructor(name: string) {
    this.name = name;
    this.div = find(name)
  }

  // Check if a point is inside the panel.
  inside(point: Vec2f): boolean {
    return point.x() >= this.x && point.x() < this.x + this.width &&
      point.y() >= this.y && point.y() < this.y + this.height;
  }

  // Transform a point from the canvas to the map coordinate system.
  transformPoint(point: Vec2f): Vec2f {
    const m = Mat3f.translate(this.x + this.width / 2, this.y + this.height / 2)
      .mul(Mat3f.scale(this.zoomLevel, this.zoomLevel))
      .mul(Mat3f.translate(this.panPos.x(), this.panPos.y()));
    return m.inverse().transform(point);
  }

  // Make the panel focus on a specific position in the panel.
  focusPos(x: number, y: number, zoomLevel: number) {
    this.panPos = new Vec2f(
      -x,
      -y
    );
    this.zoomLevel = zoomLevel;
  }

  // Update the pan and zoom level based on the mouse position and scroll delta.
  updatePanAndZoom(): boolean {

    if (ui.mouseClick) {
      this.isPanning = true;
    }
    if (!ui.mouseDown) {
      this.isPanning = false;
    }

    if (this.isPanning && ui.mousePos.sub(ui.lastMousePos).length() > 1) {
      const lastMousePoint = this.transformPoint(ui.lastMousePos);
      const newMousePoint = this.transformPoint(ui.mousePos);
      this.panPos = this.panPos.add(newMousePoint.sub(lastMousePoint));
      ui.lastMousePos = ui.mousePos;
      return true;
    }

    if (ui.scrollDelta !== 0) {
      const oldMousePoint = this.transformPoint(ui.mousePos);
      this.zoomLevel = this.zoomLevel + ui.scrollDelta / Common.SCROLL_ZOOM_FACTOR;
      this.zoomLevel = Math.max(Math.min(this.zoomLevel, Common.MAX_ZOOM_LEVEL), Common.MIN_ZOOM_LEVEL);
      const newMousePoint = this.transformPoint(ui.mousePos);
      if (oldMousePoint != null && newMousePoint != null) {
        this.panPos = this.panPos.add(newMousePoint.sub(oldMousePoint));
      }
      ui.scrollDelta = 0;
      return true;
    }
    return false;
  }

  // Update the div position and size.
  updateDiv() {
    this.div.style.top = this.y + 'px';
    this.div.style.left = this.x + 'px';
    this.div.style.width = this.width + 'px';
    this.div.style.height = this.height + 'px';
  }
}
