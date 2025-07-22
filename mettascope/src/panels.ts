import { Vec2f, Mat3f } from './vector_math.js'
import * as Common from './common.js'
import { ui } from './common.js'
import { find } from './htmlutils.js'

export class Rect {
  public x: number = 0
  public y: number = 0
  public width: number = 0
  public height: number = 0
}

/** A main UI panel. */
export class PanelInfo {
  public x: number = 0
  public y: number = 0
  public width: number = 0
  public height: number = 0
  public name: string = ''
  public isPanning: boolean = false
  public panPos: Vec2f = new Vec2f(0, 0)
  public zoomLevel: number = Common.DEFAULT_ZOOM_LEVEL
  public div: HTMLElement

  constructor(name: string) {
    this.name = name
    this.div = find(name)
  }

  /** Checks if a point is inside the panel. */
  inside(point: Vec2f): boolean {
    return (
      point.x() >= this.x && point.x() < this.x + this.width && point.y() >= this.y && point.y() < this.y + this.height
    )
  }

  /** Gets the transformation matrix for the panel. */
  transform(): Mat3f {
    const rect = this.rectInner()
    return Mat3f.scale(1 / ui.dpr, 1 / ui.dpr)
      .mul(Mat3f.translate(rect.x + rect.width / 2, rect.y + rect.height / 2))
      .mul(Mat3f.scale(this.zoomLevel, this.zoomLevel))
      .mul(Mat3f.translate(this.panPos.x(), this.panPos.y()))
  }

  /** Transforms a point from the outer coordinate system to the panel's inner coordinate system. */
  transformOuter(point: Vec2f): Vec2f {
    return this.transform().inverse().transform(point)
  }

  /** Transforms a point from the panel's inner coordinate system to the outer coordinate system. */
  transformInner(point: Vec2f): Vec2f {
    return this.transform().transform(point)
  }

  rectInner(): Rect {
    return {
      x: this.x * ui.dpr,
      y: this.y * ui.dpr,
      width: this.width * ui.dpr,
      height: this.height * ui.dpr,
    }
  }

  /** Makes the panel focus on a specific position in the panel. */
  focusPos(x: number, y: number, zoomLevel: number) {
    this.panPos = new Vec2f(-x, -y)
    this.zoomLevel = zoomLevel * ui.dpr
  }

  /** Updates the pan and zoom level based on the mouse position and scroll delta. */
  updatePanAndZoom(): boolean {
    if (!ui.mouseTargets.includes(this.name) || ui.dragging != '') {
      return false
    }

    // apply zoom at a focal point.
    const applyZoom = (focalPoint: Vec2f, zoomDelta: number) => {
      const oldPoint = this.transformOuter(focalPoint)
      this.zoomLevel = this.zoomLevel * Math.pow(1 - Common.ZOOM_SENSITIVITY, zoomDelta)
      this.zoomLevel = Math.max(Math.min(this.zoomLevel, Common.MAX_ZOOM_LEVEL), Common.MIN_ZOOM_LEVEL)
      const newPoint = this.transformOuter(focalPoint)
      if (oldPoint != null && newPoint != null) {
        this.panPos = this.panPos.add(newPoint.sub(oldPoint))
      }
    }

    if (ui.mouseClick) {
      this.isPanning = true
    }
    if (!ui.mouseDown) {
      this.isPanning = false
    }

    // Handle pinch-to-zoom gestures
    if (ui.isPinching && ui.touches.length === 2) {
      // Calculate the center point between the two touches
      const touch1 = new Vec2f(ui.touches[0].clientX, ui.touches[0].clientY)
      const touch2 = new Vec2f(ui.touches[1].clientX, ui.touches[1].clientY)
      const center = touch1.add(touch2).mul(0.5)
      const distance = touch1.sub(touch2).length()

      // Process zoom smoothly with minimal threshold for fluid movement
      if (ui.lastPinchDistance > 20 && Math.abs(distance - ui.lastPinchDistance) > 1) {
        // Calculate zoom delta (similar to scroll wheel) - smooth and responsive
        const zoomRatio = distance / ui.lastPinchDistance
        const scrollEquivalent = (zoomRatio - 1.0) * 400 // Convert to scroll-like delta

        applyZoom(center, scrollEquivalent)
      }

      // Always update distance for smooth continuous zooming
      if (distance > 20) {
        ui.lastPinchDistance = distance
      }

      // Allow panning the camera while pinching
      const centerDelta = center.sub(ui.lastPinchCenter)
      if (centerDelta.length() > 0.5) {
        // Very low threshold for smooth panning
        const lastCenterPoint = this.transformOuter(ui.lastPinchCenter)
        const newCenterPoint = this.transformOuter(center)
        if (lastCenterPoint != null && newCenterPoint != null) {
          this.panPos = this.panPos.add(newCenterPoint.sub(lastCenterPoint))
        }
      }

      // Always update center for next frame
      ui.lastPinchCenter = center
      return true
    }

    if (this.isPanning && ui.mousePos.sub(ui.lastMousePos).length() > 1) {
      const lastMousePoint = this.transformOuter(ui.lastMousePos)
      const newMousePoint = this.transformOuter(ui.mousePos)
      this.panPos = this.panPos.add(newMousePoint.sub(lastMousePoint))
      ui.lastMousePos = ui.mousePos
      return true
    }

    if (ui.scrollDelta !== 0) {
      applyZoom(ui.mousePos, ui.scrollDelta)
      ui.scrollDelta = 0
      return true
    }
    return false
  }

  /** Updates the div's position and size. */
  updateDiv() {
    this.div.style.top = this.y + 'px'
    this.div.style.left = this.x + 'px'
    this.div.style.width = this.width + 'px'
    this.div.style.height = this.height + 'px'
  }
}
