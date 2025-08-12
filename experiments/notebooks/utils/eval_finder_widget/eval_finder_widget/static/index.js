function hT(U) {
  return U && U.__esModule && Object.prototype.hasOwnProperty.call(U, "default") ? U.default : U;
}
var lS = { exports: {} }, ml = lS.exports = {}, Xi, Qi;
function L0() {
  throw new Error("setTimeout has not been defined");
}
function V0() {
  throw new Error("clearTimeout has not been defined");
}
(function() {
  try {
    typeof setTimeout == "function" ? Xi = setTimeout : Xi = L0;
  } catch {
    Xi = L0;
  }
  try {
    typeof clearTimeout == "function" ? Qi = clearTimeout : Qi = V0;
  } catch {
    Qi = V0;
  }
})();
function aS(U) {
  if (Xi === setTimeout)
    return setTimeout(U, 0);
  if ((Xi === L0 || !Xi) && setTimeout)
    return Xi = setTimeout, setTimeout(U, 0);
  try {
    return Xi(U, 0);
  } catch {
    try {
      return Xi.call(null, U, 0);
    } catch {
      return Xi.call(this, U, 0);
    }
  }
}
function yT(U) {
  if (Qi === clearTimeout)
    return clearTimeout(U);
  if ((Qi === V0 || !Qi) && clearTimeout)
    return Qi = clearTimeout, clearTimeout(U);
  try {
    return Qi(U);
  } catch {
    try {
      return Qi.call(null, U);
    } catch {
      return Qi.call(this, U);
    }
  }
}
var Fc = [], Uh = !1, ur, Tg = -1;
function mT() {
  !Uh || !ur || (Uh = !1, ur.length ? Fc = ur.concat(Fc) : Tg = -1, Fc.length && nS());
}
function nS() {
  if (!Uh) {
    var U = aS(mT);
    Uh = !0;
    for (var W = Fc.length; W; ) {
      for (ur = Fc, Fc = []; ++Tg < W; )
        ur && ur[Tg].run();
      Tg = -1, W = Fc.length;
    }
    ur = null, Uh = !1, yT(U);
  }
}
ml.nextTick = function(U) {
  var W = new Array(arguments.length - 1);
  if (arguments.length > 1)
    for (var ge = 1; ge < arguments.length; ge++)
      W[ge - 1] = arguments[ge];
  Fc.push(new uS(U, W)), Fc.length === 1 && !Uh && aS(nS);
};
function uS(U, W) {
  this.fun = U, this.array = W;
}
uS.prototype.run = function() {
  this.fun.apply(null, this.array);
};
ml.title = "browser";
ml.browser = !0;
ml.env = {};
ml.argv = [];
ml.version = "";
ml.versions = {};
function Ic() {
}
ml.on = Ic;
ml.addListener = Ic;
ml.once = Ic;
ml.off = Ic;
ml.removeListener = Ic;
ml.removeAllListeners = Ic;
ml.emit = Ic;
ml.prependListener = Ic;
ml.prependOnceListener = Ic;
ml.listeners = function(U) {
  return [];
};
ml.binding = function(U) {
  throw new Error("process.binding is not supported");
};
ml.cwd = function() {
  return "/";
};
ml.chdir = function(U) {
  throw new Error("process.chdir is not supported");
};
ml.umask = function() {
  return 0;
};
var pT = lS.exports;
const It = /* @__PURE__ */ hT(pT);
var pg = { exports: {} }, Tp = {};
/**
 * @license React
 * react-jsx-runtime.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var Lb;
function vT() {
  if (Lb) return Tp;
  Lb = 1;
  var U = Symbol.for("react.transitional.element"), W = Symbol.for("react.fragment");
  function ge(_, he, Ee) {
    var Qe = null;
    if (Ee !== void 0 && (Qe = "" + Ee), he.key !== void 0 && (Qe = "" + he.key), "key" in he) {
      Ee = {};
      for (var et in he)
        et !== "key" && (Ee[et] = he[et]);
    } else Ee = he;
    return he = Ee.ref, {
      $$typeof: U,
      type: _,
      key: Qe,
      ref: he !== void 0 ? he : null,
      props: Ee
    };
  }
  return Tp.Fragment = W, Tp.jsx = ge, Tp.jsxs = ge, Tp;
}
var Ep = {}, vg = { exports: {} }, Ie = {}, Vb;
function gT() {
  if (Vb) return Ie;
  Vb = 1;
  var U = Symbol.for("react.transitional.element"), W = Symbol.for("react.portal"), ge = Symbol.for("react.fragment"), _ = Symbol.for("react.strict_mode"), he = Symbol.for("react.profiler"), Ee = Symbol.for("react.consumer"), Qe = Symbol.for("react.context"), et = Symbol.for("react.forward_ref"), x = Symbol.for("react.suspense"), w = Symbol.for("react.memo"), te = Symbol.for("react.lazy"), G = Symbol.iterator;
  function z(g) {
    return g === null || typeof g != "object" ? null : (g = G && g[G] || g["@@iterator"], typeof g == "function" ? g : null);
  }
  var ae = {
    isMounted: function() {
      return !1;
    },
    enqueueForceUpdate: function() {
    },
    enqueueReplaceState: function() {
    },
    enqueueSetState: function() {
    }
  }, je = Object.assign, At = {};
  function ke(g, j, J) {
    this.props = g, this.context = j, this.refs = At, this.updater = J || ae;
  }
  ke.prototype.isReactComponent = {}, ke.prototype.setState = function(g, j) {
    if (typeof g != "object" && typeof g != "function" && g != null)
      throw Error(
        "takes an object of state variables to update or a function which returns an object of state variables."
      );
    this.updater.enqueueSetState(this, g, j, "setState");
  }, ke.prototype.forceUpdate = function(g) {
    this.updater.enqueueForceUpdate(this, g, "forceUpdate");
  };
  function nt() {
  }
  nt.prototype = ke.prototype;
  function el(g, j, J) {
    this.props = g, this.context = j, this.refs = At, this.updater = J || ae;
  }
  var it = el.prototype = new nt();
  it.constructor = el, je(it, ke.prototype), it.isPureReactComponent = !0;
  var Nt = Array.isArray, se = { H: null, A: null, T: null, S: null, V: null }, Ue = Object.prototype.hasOwnProperty;
  function De(g, j, J, I, ce, Oe) {
    return J = Oe.ref, {
      $$typeof: U,
      type: g,
      key: j,
      ref: J !== void 0 ? J : null,
      props: Oe
    };
  }
  function bt(g, j) {
    return De(
      g.type,
      j,
      void 0,
      void 0,
      void 0,
      g.props
    );
  }
  function re(g) {
    return typeof g == "object" && g !== null && g.$$typeof === U;
  }
  function Rt(g) {
    var j = { "=": "=0", ":": "=2" };
    return "$" + g.replace(/[=:]/g, function(J) {
      return j[J];
    });
  }
  var Se = /\/+/g;
  function ze(g, j) {
    return typeof g == "object" && g !== null && g.key != null ? Rt("" + g.key) : j.toString(36);
  }
  function Ot() {
  }
  function Gt(g) {
    switch (g.status) {
      case "fulfilled":
        return g.value;
      case "rejected":
        throw g.reason;
      default:
        switch (typeof g.status == "string" ? g.then(Ot, Ot) : (g.status = "pending", g.then(
          function(j) {
            g.status === "pending" && (g.status = "fulfilled", g.value = j);
          },
          function(j) {
            g.status === "pending" && (g.status = "rejected", g.reason = j);
          }
        )), g.status) {
          case "fulfilled":
            return g.value;
          case "rejected":
            throw g.reason;
        }
    }
    throw g;
  }
  function pt(g, j, J, I, ce) {
    var Oe = typeof g;
    (Oe === "undefined" || Oe === "boolean") && (g = null);
    var oe = !1;
    if (g === null) oe = !0;
    else
      switch (Oe) {
        case "bigint":
        case "string":
        case "number":
          oe = !0;
          break;
        case "object":
          switch (g.$$typeof) {
            case U:
            case W:
              oe = !0;
              break;
            case te:
              return oe = g._init, pt(
                oe(g._payload),
                j,
                J,
                I,
                ce
              );
          }
      }
    if (oe)
      return ce = ce(g), oe = I === "" ? "." + ze(g, 0) : I, Nt(ce) ? (J = "", oe != null && (J = oe.replace(Se, "$&/") + "/"), pt(ce, j, J, "", function(wt) {
        return wt;
      })) : ce != null && (re(ce) && (ce = bt(
        ce,
        J + (ce.key == null || g && g.key === ce.key ? "" : ("" + ce.key).replace(
          Se,
          "$&/"
        ) + "/") + oe
      )), j.push(ce)), 1;
    oe = 0;
    var il = I === "" ? "." : I + ":";
    if (Nt(g))
      for (var He = 0; He < g.length; He++)
        I = g[He], Oe = il + ze(I, He), oe += pt(
          I,
          j,
          J,
          Oe,
          ce
        );
    else if (He = z(g), typeof He == "function")
      for (g = He.call(g), He = 0; !(I = g.next()).done; )
        I = I.value, Oe = il + ze(I, He++), oe += pt(
          I,
          j,
          J,
          Oe,
          ce
        );
    else if (Oe === "object") {
      if (typeof g.then == "function")
        return pt(
          Gt(g),
          j,
          J,
          I,
          ce
        );
      throw j = String(g), Error(
        "Objects are not valid as a React child (found: " + (j === "[object Object]" ? "object with keys {" + Object.keys(g).join(", ") + "}" : j) + "). If you meant to render a collection of children, use an array instead."
      );
    }
    return oe;
  }
  function O(g, j, J) {
    if (g == null) return g;
    var I = [], ce = 0;
    return pt(g, I, "", "", function(Oe) {
      return j.call(J, Oe, ce++);
    }), I;
  }
  function F(g) {
    if (g._status === -1) {
      var j = g._result;
      j = j(), j.then(
        function(J) {
          (g._status === 0 || g._status === -1) && (g._status = 1, g._result = J);
        },
        function(J) {
          (g._status === 0 || g._status === -1) && (g._status = 2, g._result = J);
        }
      ), g._status === -1 && (g._status = 0, g._result = j);
    }
    if (g._status === 1) return g._result.default;
    throw g._result;
  }
  var P = typeof reportError == "function" ? reportError : function(g) {
    if (typeof window == "object" && typeof window.ErrorEvent == "function") {
      var j = new window.ErrorEvent("error", {
        bubbles: !0,
        cancelable: !0,
        message: typeof g == "object" && g !== null && typeof g.message == "string" ? String(g.message) : String(g),
        error: g
      });
      if (!window.dispatchEvent(j)) return;
    } else if (typeof It == "object" && typeof It.emit == "function") {
      It.emit("uncaughtException", g);
      return;
    }
    console.error(g);
  };
  function be() {
  }
  return Ie.Children = {
    map: O,
    forEach: function(g, j, J) {
      O(
        g,
        function() {
          j.apply(this, arguments);
        },
        J
      );
    },
    count: function(g) {
      var j = 0;
      return O(g, function() {
        j++;
      }), j;
    },
    toArray: function(g) {
      return O(g, function(j) {
        return j;
      }) || [];
    },
    only: function(g) {
      if (!re(g))
        throw Error(
          "React.Children.only expected to receive a single React element child."
        );
      return g;
    }
  }, Ie.Component = ke, Ie.Fragment = ge, Ie.Profiler = he, Ie.PureComponent = el, Ie.StrictMode = _, Ie.Suspense = x, Ie.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = se, Ie.__COMPILER_RUNTIME = {
    __proto__: null,
    c: function(g) {
      return se.H.useMemoCache(g);
    }
  }, Ie.cache = function(g) {
    return function() {
      return g.apply(null, arguments);
    };
  }, Ie.cloneElement = function(g, j, J) {
    if (g == null)
      throw Error(
        "The argument must be a React element, but you passed " + g + "."
      );
    var I = je({}, g.props), ce = g.key, Oe = void 0;
    if (j != null)
      for (oe in j.ref !== void 0 && (Oe = void 0), j.key !== void 0 && (ce = "" + j.key), j)
        !Ue.call(j, oe) || oe === "key" || oe === "__self" || oe === "__source" || oe === "ref" && j.ref === void 0 || (I[oe] = j[oe]);
    var oe = arguments.length - 2;
    if (oe === 1) I.children = J;
    else if (1 < oe) {
      for (var il = Array(oe), He = 0; He < oe; He++)
        il[He] = arguments[He + 2];
      I.children = il;
    }
    return De(g.type, ce, void 0, void 0, Oe, I);
  }, Ie.createContext = function(g) {
    return g = {
      $$typeof: Qe,
      _currentValue: g,
      _currentValue2: g,
      _threadCount: 0,
      Provider: null,
      Consumer: null
    }, g.Provider = g, g.Consumer = {
      $$typeof: Ee,
      _context: g
    }, g;
  }, Ie.createElement = function(g, j, J) {
    var I, ce = {}, Oe = null;
    if (j != null)
      for (I in j.key !== void 0 && (Oe = "" + j.key), j)
        Ue.call(j, I) && I !== "key" && I !== "__self" && I !== "__source" && (ce[I] = j[I]);
    var oe = arguments.length - 2;
    if (oe === 1) ce.children = J;
    else if (1 < oe) {
      for (var il = Array(oe), He = 0; He < oe; He++)
        il[He] = arguments[He + 2];
      ce.children = il;
    }
    if (g && g.defaultProps)
      for (I in oe = g.defaultProps, oe)
        ce[I] === void 0 && (ce[I] = oe[I]);
    return De(g, Oe, void 0, void 0, null, ce);
  }, Ie.createRef = function() {
    return { current: null };
  }, Ie.forwardRef = function(g) {
    return { $$typeof: et, render: g };
  }, Ie.isValidElement = re, Ie.lazy = function(g) {
    return {
      $$typeof: te,
      _payload: { _status: -1, _result: g },
      _init: F
    };
  }, Ie.memo = function(g, j) {
    return {
      $$typeof: w,
      type: g,
      compare: j === void 0 ? null : j
    };
  }, Ie.startTransition = function(g) {
    var j = se.T, J = {};
    se.T = J;
    try {
      var I = g(), ce = se.S;
      ce !== null && ce(J, I), typeof I == "object" && I !== null && typeof I.then == "function" && I.then(be, P);
    } catch (Oe) {
      P(Oe);
    } finally {
      se.T = j;
    }
  }, Ie.unstable_useCacheRefresh = function() {
    return se.H.useCacheRefresh();
  }, Ie.use = function(g) {
    return se.H.use(g);
  }, Ie.useActionState = function(g, j, J) {
    return se.H.useActionState(g, j, J);
  }, Ie.useCallback = function(g, j) {
    return se.H.useCallback(g, j);
  }, Ie.useContext = function(g) {
    return se.H.useContext(g);
  }, Ie.useDebugValue = function() {
  }, Ie.useDeferredValue = function(g, j) {
    return se.H.useDeferredValue(g, j);
  }, Ie.useEffect = function(g, j, J) {
    var I = se.H;
    if (typeof J == "function")
      throw Error(
        "useEffect CRUD overload is not enabled in this build of React."
      );
    return I.useEffect(g, j);
  }, Ie.useId = function() {
    return se.H.useId();
  }, Ie.useImperativeHandle = function(g, j, J) {
    return se.H.useImperativeHandle(g, j, J);
  }, Ie.useInsertionEffect = function(g, j) {
    return se.H.useInsertionEffect(g, j);
  }, Ie.useLayoutEffect = function(g, j) {
    return se.H.useLayoutEffect(g, j);
  }, Ie.useMemo = function(g, j) {
    return se.H.useMemo(g, j);
  }, Ie.useOptimistic = function(g, j) {
    return se.H.useOptimistic(g, j);
  }, Ie.useReducer = function(g, j, J) {
    return se.H.useReducer(g, j, J);
  }, Ie.useRef = function(g) {
    return se.H.useRef(g);
  }, Ie.useState = function(g) {
    return se.H.useState(g);
  }, Ie.useSyncExternalStore = function(g, j, J) {
    return se.H.useSyncExternalStore(
      g,
      j,
      J
    );
  }, Ie.useTransition = function() {
    return se.H.useTransition();
  }, Ie.version = "19.1.1", Ie;
}
var Op = { exports: {} };
Op.exports;
var Xb;
function bT() {
  return Xb || (Xb = 1, function(U, W) {
    It.env.NODE_ENV !== "production" && function() {
      function ge(m, D) {
        Object.defineProperty(Ee.prototype, m, {
          get: function() {
            console.warn(
              "%s(...) is deprecated in plain JavaScript React classes. %s",
              D[0],
              D[1]
            );
          }
        });
      }
      function _(m) {
        return m === null || typeof m != "object" ? null : (m = Dn && m[Dn] || m["@@iterator"], typeof m == "function" ? m : null);
      }
      function he(m, D) {
        m = (m = m.constructor) && (m.displayName || m.name) || "ReactClass";
        var le = m + "." + D;
        Zi[le] || (console.error(
          "Can't call %s on a component that is not yet mounted. This is a no-op, but it might indicate a bug in your application. Instead, assign to `this.state` directly or define a `state = {};` class property with the desired state in the %s component.",
          D,
          m
        ), Zi[le] = !0);
      }
      function Ee(m, D, le) {
        this.props = m, this.context = D, this.refs = bf, this.updater = le || zn;
      }
      function Qe() {
      }
      function et(m, D, le) {
        this.props = m, this.context = D, this.refs = bf, this.updater = le || zn;
      }
      function x(m) {
        return "" + m;
      }
      function w(m) {
        try {
          x(m);
          var D = !1;
        } catch {
          D = !0;
        }
        if (D) {
          D = console;
          var le = D.error, ue = typeof Symbol == "function" && Symbol.toStringTag && m[Symbol.toStringTag] || m.constructor.name || "Object";
          return le.call(
            D,
            "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
            ue
          ), x(m);
        }
      }
      function te(m) {
        if (m == null) return null;
        if (typeof m == "function")
          return m.$$typeof === ir ? null : m.displayName || m.name || null;
        if (typeof m == "string") return m;
        switch (m) {
          case g:
            return "Fragment";
          case J:
            return "Profiler";
          case j:
            return "StrictMode";
          case oe:
            return "Suspense";
          case il:
            return "SuspenseList";
          case na:
            return "Activity";
        }
        if (typeof m == "object")
          switch (typeof m.tag == "number" && console.error(
            "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
          ), m.$$typeof) {
            case be:
              return "Portal";
            case ce:
              return (m.displayName || "Context") + ".Provider";
            case I:
              return (m._context.displayName || "Context") + ".Consumer";
            case Oe:
              var D = m.render;
              return m = m.displayName, m || (m = D.displayName || D.name || "", m = m !== "" ? "ForwardRef(" + m + ")" : "ForwardRef"), m;
            case He:
              return D = m.displayName || null, D !== null ? D : te(m.type) || "Memo";
            case wt:
              D = m._payload, m = m._init;
              try {
                return te(m(D));
              } catch {
              }
          }
        return null;
      }
      function G(m) {
        if (m === g) return "<>";
        if (typeof m == "object" && m !== null && m.$$typeof === wt)
          return "<...>";
        try {
          var D = te(m);
          return D ? "<" + D + ">" : "<...>";
        } catch {
          return "<...>";
        }
      }
      function z() {
        var m = Ke.A;
        return m === null ? null : m.getOwner();
      }
      function ae() {
        return Error("react-stack-top-frame");
      }
      function je(m) {
        if (Mn.call(m, "key")) {
          var D = Object.getOwnPropertyDescriptor(m, "key").get;
          if (D && D.isReactWarning) return !1;
        }
        return m.key !== void 0;
      }
      function At(m, D) {
        function le() {
          gu || (gu = !0, console.error(
            "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
            D
          ));
        }
        le.isReactWarning = !0, Object.defineProperty(m, "key", {
          get: le,
          configurable: !0
        });
      }
      function ke() {
        var m = te(this.type);
        return Sf[m] || (Sf[m] = !0, console.error(
          "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
        )), m = this.props.ref, m !== void 0 ? m : null;
      }
      function nt(m, D, le, ue, ve, Ne, Ye, ct) {
        return le = Ne.ref, m = {
          $$typeof: P,
          type: m,
          key: D,
          props: Ne,
          _owner: ve
        }, (le !== void 0 ? le : null) !== null ? Object.defineProperty(m, "ref", {
          enumerable: !1,
          get: ke
        }) : Object.defineProperty(m, "ref", { enumerable: !1, value: null }), m._store = {}, Object.defineProperty(m._store, "validated", {
          configurable: !1,
          enumerable: !1,
          writable: !0,
          value: 0
        }), Object.defineProperty(m, "_debugInfo", {
          configurable: !1,
          enumerable: !1,
          writable: !0,
          value: null
        }), Object.defineProperty(m, "_debugStack", {
          configurable: !1,
          enumerable: !1,
          writable: !0,
          value: Ye
        }), Object.defineProperty(m, "_debugTask", {
          configurable: !1,
          enumerable: !1,
          writable: !0,
          value: ct
        }), Object.freeze && (Object.freeze(m.props), Object.freeze(m)), m;
      }
      function el(m, D) {
        return D = nt(
          m.type,
          D,
          void 0,
          void 0,
          m._owner,
          m.props,
          m._debugStack,
          m._debugTask
        ), m._store && (D._store.validated = m._store.validated), D;
      }
      function it(m) {
        return typeof m == "object" && m !== null && m.$$typeof === P;
      }
      function Nt(m) {
        var D = { "=": "=0", ":": "=2" };
        return "$" + m.replace(/[=:]/g, function(le) {
          return D[le];
        });
      }
      function se(m, D) {
        return typeof m == "object" && m !== null && m.key != null ? (w(m.key), Nt("" + m.key)) : D.toString(36);
      }
      function Ue() {
      }
      function De(m) {
        switch (m.status) {
          case "fulfilled":
            return m.value;
          case "rejected":
            throw m.reason;
          default:
            switch (typeof m.status == "string" ? m.then(Ue, Ue) : (m.status = "pending", m.then(
              function(D) {
                m.status === "pending" && (m.status = "fulfilled", m.value = D);
              },
              function(D) {
                m.status === "pending" && (m.status = "rejected", m.reason = D);
              }
            )), m.status) {
              case "fulfilled":
                return m.value;
              case "rejected":
                throw m.reason;
            }
        }
        throw m;
      }
      function bt(m, D, le, ue, ve) {
        var Ne = typeof m;
        (Ne === "undefined" || Ne === "boolean") && (m = null);
        var Ye = !1;
        if (m === null) Ye = !0;
        else
          switch (Ne) {
            case "bigint":
            case "string":
            case "number":
              Ye = !0;
              break;
            case "object":
              switch (m.$$typeof) {
                case P:
                case be:
                  Ye = !0;
                  break;
                case wt:
                  return Ye = m._init, bt(
                    Ye(m._payload),
                    D,
                    le,
                    ue,
                    ve
                  );
              }
          }
        if (Ye) {
          Ye = m, ve = ve(Ye);
          var ct = ue === "" ? "." + se(Ye, 0) : ue;
          return Iu(ve) ? (le = "", ct != null && (le = ct.replace(Dl, "$&/") + "/"), bt(ve, D, le, "", function(ll) {
            return ll;
          })) : ve != null && (it(ve) && (ve.key != null && (Ye && Ye.key === ve.key || w(ve.key)), le = el(
            ve,
            le + (ve.key == null || Ye && Ye.key === ve.key ? "" : ("" + ve.key).replace(
              Dl,
              "$&/"
            ) + "/") + ct
          ), ue !== "" && Ye != null && it(Ye) && Ye.key == null && Ye._store && !Ye._store.validated && (le._store.validated = 2), ve = le), D.push(ve)), 1;
        }
        if (Ye = 0, ct = ue === "" ? "." : ue + ":", Iu(m))
          for (var Be = 0; Be < m.length; Be++)
            ue = m[Be], Ne = ct + se(ue, Be), Ye += bt(
              ue,
              D,
              le,
              Ne,
              ve
            );
        else if (Be = _(m), typeof Be == "function")
          for (Be === m.entries && (Ba || console.warn(
            "Using Maps as children is not supported. Use an array of keyed ReactElements instead."
          ), Ba = !0), m = Be.call(m), Be = 0; !(ue = m.next()).done; )
            ue = ue.value, Ne = ct + se(ue, Be++), Ye += bt(
              ue,
              D,
              le,
              Ne,
              ve
            );
        else if (Ne === "object") {
          if (typeof m.then == "function")
            return bt(
              De(m),
              D,
              le,
              ue,
              ve
            );
          throw D = String(m), Error(
            "Objects are not valid as a React child (found: " + (D === "[object Object]" ? "object with keys {" + Object.keys(m).join(", ") + "}" : D) + "). If you meant to render a collection of children, use an array instead."
          );
        }
        return Ye;
      }
      function re(m, D, le) {
        if (m == null) return m;
        var ue = [], ve = 0;
        return bt(m, ue, "", "", function(Ne) {
          return D.call(le, Ne, ve++);
        }), ue;
      }
      function Rt(m) {
        if (m._status === -1) {
          var D = m._result;
          D = D(), D.then(
            function(le) {
              (m._status === 0 || m._status === -1) && (m._status = 1, m._result = le);
            },
            function(le) {
              (m._status === 0 || m._status === -1) && (m._status = 2, m._result = le);
            }
          ), m._status === -1 && (m._status = 0, m._result = D);
        }
        if (m._status === 1)
          return D = m._result, D === void 0 && console.error(
            `lazy: Expected the result of a dynamic import() call. Instead received: %s

Your code should look like: 
  const MyComponent = lazy(() => import('./MyComponent'))

Did you accidentally put curly braces around the import?`,
            D
          ), "default" in D || console.error(
            `lazy: Expected the result of a dynamic import() call. Instead received: %s

Your code should look like: 
  const MyComponent = lazy(() => import('./MyComponent'))`,
            D
          ), D.default;
        throw m._result;
      }
      function Se() {
        var m = Ke.H;
        return m === null && console.error(
          `Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:
1. You might have mismatching versions of React and the renderer (such as React DOM)
2. You might be breaking the Rules of Hooks
3. You might have more than one copy of React in the same app
See https://react.dev/link/invalid-hook-call for tips about how to debug and fix this problem.`
        ), m;
      }
      function ze() {
      }
      function Ot(m) {
        if (lo === null)
          try {
            var D = ("require" + Math.random()).slice(0, 7);
            lo = (U && U[D]).call(
              U,
              "timers"
            ).setImmediate;
          } catch {
            lo = function(ue) {
              Tf === !1 && (Tf = !0, typeof MessageChannel > "u" && console.error(
                "This browser does not have a MessageChannel implementation, so enqueuing tasks via await act(async () => ...) will fail. Please file an issue at https://github.com/facebook/react/issues if you encounter this warning."
              ));
              var ve = new MessageChannel();
              ve.port1.onmessage = ue, ve.port2.postMessage(void 0);
            };
          }
        return lo(m);
      }
      function Gt(m) {
        return 1 < m.length && typeof AggregateError == "function" ? new AggregateError(m) : m[0];
      }
      function pt(m, D) {
        D !== nn - 1 && console.error(
          "You seem to have overlapping act() calls, this is not supported. Be sure to await previous act() calls before making a new one. "
        ), nn = D;
      }
      function O(m, D, le) {
        var ue = Ke.actQueue;
        if (ue !== null)
          if (ue.length !== 0)
            try {
              F(ue), Ot(function() {
                return O(m, D, le);
              });
              return;
            } catch (ve) {
              Ke.thrownErrors.push(ve);
            }
          else Ke.actQueue = null;
        0 < Ke.thrownErrors.length ? (ue = Gt(Ke.thrownErrors), Ke.thrownErrors.length = 0, le(ue)) : D(m);
      }
      function F(m) {
        if (!zl) {
          zl = !0;
          var D = 0;
          try {
            for (; D < m.length; D++) {
              var le = m[D];
              do {
                Ke.didUsePromise = !1;
                var ue = le(!1);
                if (ue !== null) {
                  if (Ke.didUsePromise) {
                    m[D] = le, m.splice(0, D);
                    return;
                  }
                  le = ue;
                } else break;
              } while (!0);
            }
            m.length = 0;
          } catch (ve) {
            m.splice(0, D + 1), Ke.thrownErrors.push(ve);
          } finally {
            zl = !1;
          }
        }
      }
      typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
      var P = Symbol.for("react.transitional.element"), be = Symbol.for("react.portal"), g = Symbol.for("react.fragment"), j = Symbol.for("react.strict_mode"), J = Symbol.for("react.profiler"), I = Symbol.for("react.consumer"), ce = Symbol.for("react.context"), Oe = Symbol.for("react.forward_ref"), oe = Symbol.for("react.suspense"), il = Symbol.for("react.suspense_list"), He = Symbol.for("react.memo"), wt = Symbol.for("react.lazy"), na = Symbol.for("react.activity"), Dn = Symbol.iterator, Zi = {}, zn = {
        isMounted: function() {
          return !1;
        },
        enqueueForceUpdate: function(m) {
          he(m, "forceUpdate");
        },
        enqueueReplaceState: function(m) {
          he(m, "replaceState");
        },
        enqueueSetState: function(m) {
          he(m, "setState");
        }
      }, Pc = Object.assign, bf = {};
      Object.freeze(bf), Ee.prototype.isReactComponent = {}, Ee.prototype.setState = function(m, D) {
        if (typeof m != "object" && typeof m != "function" && m != null)
          throw Error(
            "takes an object of state variables to update or a function which returns an object of state variables."
          );
        this.updater.enqueueSetState(this, m, D, "setState");
      }, Ee.prototype.forceUpdate = function(m) {
        this.updater.enqueueForceUpdate(this, m, "forceUpdate");
      };
      var tl = {
        isMounted: [
          "isMounted",
          "Instead, make sure to clean up subscriptions and pending requests in componentWillUnmount to prevent memory leaks."
        ],
        replaceState: [
          "replaceState",
          "Refactor your code to use setState instead (see https://github.com/facebook/react/issues/3236)."
        ]
      }, pl;
      for (pl in tl)
        tl.hasOwnProperty(pl) && ge(pl, tl[pl]);
      Qe.prototype = Ee.prototype, tl = et.prototype = new Qe(), tl.constructor = et, Pc(tl, Ee.prototype), tl.isPureReactComponent = !0;
      var Iu = Array.isArray, ir = Symbol.for("react.client.reference"), Ke = {
        H: null,
        A: null,
        T: null,
        S: null,
        V: null,
        actQueue: null,
        isBatchingLegacy: !1,
        didScheduleLegacyUpdate: !1,
        didUsePromise: !1,
        thrownErrors: [],
        getCurrentStack: null,
        recentlyCreatedOwnerStacks: 0
      }, Mn = Object.prototype.hasOwnProperty, eo = console.createTask ? console.createTask : function() {
        return null;
      };
      tl = {
        react_stack_bottom_frame: function(m) {
          return m();
        }
      };
      var gu, cr, Sf = {}, Pu = tl.react_stack_bottom_frame.bind(
        tl,
        ae
      )(), Ol = eo(G(ae)), Ba = !1, Dl = /\/+/g, to = typeof reportError == "function" ? reportError : function(m) {
        if (typeof window == "object" && typeof window.ErrorEvent == "function") {
          var D = new window.ErrorEvent("error", {
            bubbles: !0,
            cancelable: !0,
            message: typeof m == "object" && m !== null && typeof m.message == "string" ? String(m.message) : String(m),
            error: m
          });
          if (!window.dispatchEvent(D)) return;
        } else if (typeof It == "object" && typeof It.emit == "function") {
          It.emit("uncaughtException", m);
          return;
        }
        console.error(m);
      }, Tf = !1, lo = null, nn = 0, ua = !1, zl = !1, un = typeof queueMicrotask == "function" ? function(m) {
        queueMicrotask(function() {
          return queueMicrotask(m);
        });
      } : Ot;
      tl = Object.freeze({
        __proto__: null,
        c: function(m) {
          return Se().useMemoCache(m);
        }
      }), W.Children = {
        map: re,
        forEach: function(m, D, le) {
          re(
            m,
            function() {
              D.apply(this, arguments);
            },
            le
          );
        },
        count: function(m) {
          var D = 0;
          return re(m, function() {
            D++;
          }), D;
        },
        toArray: function(m) {
          return re(m, function(D) {
            return D;
          }) || [];
        },
        only: function(m) {
          if (!it(m))
            throw Error(
              "React.Children.only expected to receive a single React element child."
            );
          return m;
        }
      }, W.Component = Ee, W.Fragment = g, W.Profiler = J, W.PureComponent = et, W.StrictMode = j, W.Suspense = oe, W.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = Ke, W.__COMPILER_RUNTIME = tl, W.act = function(m) {
        var D = Ke.actQueue, le = nn;
        nn++;
        var ue = Ke.actQueue = D !== null ? D : [], ve = !1;
        try {
          var Ne = m();
        } catch (Be) {
          Ke.thrownErrors.push(Be);
        }
        if (0 < Ke.thrownErrors.length)
          throw pt(D, le), m = Gt(Ke.thrownErrors), Ke.thrownErrors.length = 0, m;
        if (Ne !== null && typeof Ne == "object" && typeof Ne.then == "function") {
          var Ye = Ne;
          return un(function() {
            ve || ua || (ua = !0, console.error(
              "You called act(async () => ...) without await. This could lead to unexpected testing behaviour, interleaving multiple act calls and mixing their scopes. You should - await act(async () => ...);"
            ));
          }), {
            then: function(Be, ll) {
              ve = !0, Ye.then(
                function(cn) {
                  if (pt(D, le), le === 0) {
                    try {
                      F(ue), Ot(function() {
                        return O(
                          cn,
                          Be,
                          ll
                        );
                      });
                    } catch (xh) {
                      Ke.thrownErrors.push(xh);
                    }
                    if (0 < Ke.thrownErrors.length) {
                      var or = Gt(
                        Ke.thrownErrors
                      );
                      Ke.thrownErrors.length = 0, ll(or);
                    }
                  } else Be(cn);
                },
                function(cn) {
                  pt(D, le), 0 < Ke.thrownErrors.length && (cn = Gt(
                    Ke.thrownErrors
                  ), Ke.thrownErrors.length = 0), ll(cn);
                }
              );
            }
          };
        }
        var ct = Ne;
        if (pt(D, le), le === 0 && (F(ue), ue.length !== 0 && un(function() {
          ve || ua || (ua = !0, console.error(
            "A component suspended inside an `act` scope, but the `act` call was not awaited. When testing React components that depend on asynchronous data, you must await the result:\n\nawait act(() => ...)"
          ));
        }), Ke.actQueue = null), 0 < Ke.thrownErrors.length)
          throw m = Gt(Ke.thrownErrors), Ke.thrownErrors.length = 0, m;
        return {
          then: function(Be, ll) {
            ve = !0, le === 0 ? (Ke.actQueue = ue, Ot(function() {
              return O(
                ct,
                Be,
                ll
              );
            })) : Be(ct);
          }
        };
      }, W.cache = function(m) {
        return function() {
          return m.apply(null, arguments);
        };
      }, W.captureOwnerStack = function() {
        var m = Ke.getCurrentStack;
        return m === null ? null : m();
      }, W.cloneElement = function(m, D, le) {
        if (m == null)
          throw Error(
            "The argument must be a React element, but you passed " + m + "."
          );
        var ue = Pc({}, m.props), ve = m.key, Ne = m._owner;
        if (D != null) {
          var Ye;
          e: {
            if (Mn.call(D, "ref") && (Ye = Object.getOwnPropertyDescriptor(
              D,
              "ref"
            ).get) && Ye.isReactWarning) {
              Ye = !1;
              break e;
            }
            Ye = D.ref !== void 0;
          }
          Ye && (Ne = z()), je(D) && (w(D.key), ve = "" + D.key);
          for (ct in D)
            !Mn.call(D, ct) || ct === "key" || ct === "__self" || ct === "__source" || ct === "ref" && D.ref === void 0 || (ue[ct] = D[ct]);
        }
        var ct = arguments.length - 2;
        if (ct === 1) ue.children = le;
        else if (1 < ct) {
          Ye = Array(ct);
          for (var Be = 0; Be < ct; Be++)
            Ye[Be] = arguments[Be + 2];
          ue.children = Ye;
        }
        for (ue = nt(
          m.type,
          ve,
          void 0,
          void 0,
          Ne,
          ue,
          m._debugStack,
          m._debugTask
        ), ve = 2; ve < arguments.length; ve++)
          Ne = arguments[ve], it(Ne) && Ne._store && (Ne._store.validated = 1);
        return ue;
      }, W.createContext = function(m) {
        return m = {
          $$typeof: ce,
          _currentValue: m,
          _currentValue2: m,
          _threadCount: 0,
          Provider: null,
          Consumer: null
        }, m.Provider = m, m.Consumer = {
          $$typeof: I,
          _context: m
        }, m._currentRenderer = null, m._currentRenderer2 = null, m;
      }, W.createElement = function(m, D, le) {
        for (var ue = 2; ue < arguments.length; ue++) {
          var ve = arguments[ue];
          it(ve) && ve._store && (ve._store.validated = 1);
        }
        if (ue = {}, ve = null, D != null)
          for (Be in cr || !("__self" in D) || "key" in D || (cr = !0, console.warn(
            "Your app (or one of its dependencies) is using an outdated JSX transform. Update to the modern JSX transform for faster performance: https://react.dev/link/new-jsx-transform"
          )), je(D) && (w(D.key), ve = "" + D.key), D)
            Mn.call(D, Be) && Be !== "key" && Be !== "__self" && Be !== "__source" && (ue[Be] = D[Be]);
        var Ne = arguments.length - 2;
        if (Ne === 1) ue.children = le;
        else if (1 < Ne) {
          for (var Ye = Array(Ne), ct = 0; ct < Ne; ct++)
            Ye[ct] = arguments[ct + 2];
          Object.freeze && Object.freeze(Ye), ue.children = Ye;
        }
        if (m && m.defaultProps)
          for (Be in Ne = m.defaultProps, Ne)
            ue[Be] === void 0 && (ue[Be] = Ne[Be]);
        ve && At(
          ue,
          typeof m == "function" ? m.displayName || m.name || "Unknown" : m
        );
        var Be = 1e4 > Ke.recentlyCreatedOwnerStacks++;
        return nt(
          m,
          ve,
          void 0,
          void 0,
          z(),
          ue,
          Be ? Error("react-stack-top-frame") : Pu,
          Be ? eo(G(m)) : Ol
        );
      }, W.createRef = function() {
        var m = { current: null };
        return Object.seal(m), m;
      }, W.forwardRef = function(m) {
        m != null && m.$$typeof === He ? console.error(
          "forwardRef requires a render function but received a `memo` component. Instead of forwardRef(memo(...)), use memo(forwardRef(...))."
        ) : typeof m != "function" ? console.error(
          "forwardRef requires a render function but was given %s.",
          m === null ? "null" : typeof m
        ) : m.length !== 0 && m.length !== 2 && console.error(
          "forwardRef render functions accept exactly two parameters: props and ref. %s",
          m.length === 1 ? "Did you forget to use the ref parameter?" : "Any additional parameter will be undefined."
        ), m != null && m.defaultProps != null && console.error(
          "forwardRef render functions do not support defaultProps. Did you accidentally pass a React component?"
        );
        var D = { $$typeof: Oe, render: m }, le;
        return Object.defineProperty(D, "displayName", {
          enumerable: !1,
          configurable: !0,
          get: function() {
            return le;
          },
          set: function(ue) {
            le = ue, m.name || m.displayName || (Object.defineProperty(m, "name", { value: ue }), m.displayName = ue);
          }
        }), D;
      }, W.isValidElement = it, W.lazy = function(m) {
        return {
          $$typeof: wt,
          _payload: { _status: -1, _result: m },
          _init: Rt
        };
      }, W.memo = function(m, D) {
        m == null && console.error(
          "memo: The first argument must be a component. Instead received: %s",
          m === null ? "null" : typeof m
        ), D = {
          $$typeof: He,
          type: m,
          compare: D === void 0 ? null : D
        };
        var le;
        return Object.defineProperty(D, "displayName", {
          enumerable: !1,
          configurable: !0,
          get: function() {
            return le;
          },
          set: function(ue) {
            le = ue, m.name || m.displayName || (Object.defineProperty(m, "name", { value: ue }), m.displayName = ue);
          }
        }), D;
      }, W.startTransition = function(m) {
        var D = Ke.T, le = {};
        Ke.T = le, le._updatedFibers = /* @__PURE__ */ new Set();
        try {
          var ue = m(), ve = Ke.S;
          ve !== null && ve(le, ue), typeof ue == "object" && ue !== null && typeof ue.then == "function" && ue.then(ze, to);
        } catch (Ne) {
          to(Ne);
        } finally {
          D === null && le._updatedFibers && (m = le._updatedFibers.size, le._updatedFibers.clear(), 10 < m && console.warn(
            "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
          )), Ke.T = D;
        }
      }, W.unstable_useCacheRefresh = function() {
        return Se().useCacheRefresh();
      }, W.use = function(m) {
        return Se().use(m);
      }, W.useActionState = function(m, D, le) {
        return Se().useActionState(
          m,
          D,
          le
        );
      }, W.useCallback = function(m, D) {
        return Se().useCallback(m, D);
      }, W.useContext = function(m) {
        var D = Se();
        return m.$$typeof === I && console.error(
          "Calling useContext(Context.Consumer) is not supported and will cause bugs. Did you mean to call useContext(Context) instead?"
        ), D.useContext(m);
      }, W.useDebugValue = function(m, D) {
        return Se().useDebugValue(m, D);
      }, W.useDeferredValue = function(m, D) {
        return Se().useDeferredValue(m, D);
      }, W.useEffect = function(m, D, le) {
        m == null && console.warn(
          "React Hook useEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        );
        var ue = Se();
        if (typeof le == "function")
          throw Error(
            "useEffect CRUD overload is not enabled in this build of React."
          );
        return ue.useEffect(m, D);
      }, W.useId = function() {
        return Se().useId();
      }, W.useImperativeHandle = function(m, D, le) {
        return Se().useImperativeHandle(m, D, le);
      }, W.useInsertionEffect = function(m, D) {
        return m == null && console.warn(
          "React Hook useInsertionEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        ), Se().useInsertionEffect(m, D);
      }, W.useLayoutEffect = function(m, D) {
        return m == null && console.warn(
          "React Hook useLayoutEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        ), Se().useLayoutEffect(m, D);
      }, W.useMemo = function(m, D) {
        return Se().useMemo(m, D);
      }, W.useOptimistic = function(m, D) {
        return Se().useOptimistic(m, D);
      }, W.useReducer = function(m, D, le) {
        return Se().useReducer(m, D, le);
      }, W.useRef = function(m) {
        return Se().useRef(m);
      }, W.useState = function(m) {
        return Se().useState(m);
      }, W.useSyncExternalStore = function(m, D, le) {
        return Se().useSyncExternalStore(
          m,
          D,
          le
        );
      }, W.useTransition = function() {
        return Se().useTransition();
      }, W.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    }();
  }(Op, Op.exports)), Op.exports;
}
var Qb;
function Ch() {
  return Qb || (Qb = 1, It.env.NODE_ENV === "production" ? vg.exports = gT() : vg.exports = bT()), vg.exports;
}
var Zb;
function ST() {
  return Zb || (Zb = 1, It.env.NODE_ENV !== "production" && function() {
    function U(g) {
      if (g == null) return null;
      if (typeof g == "function")
        return g.$$typeof === Rt ? null : g.displayName || g.name || null;
      if (typeof g == "string") return g;
      switch (g) {
        case At:
          return "Fragment";
        case nt:
          return "Profiler";
        case ke:
          return "StrictMode";
        case se:
          return "Suspense";
        case Ue:
          return "SuspenseList";
        case re:
          return "Activity";
      }
      if (typeof g == "object")
        switch (typeof g.tag == "number" && console.error(
          "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
        ), g.$$typeof) {
          case je:
            return "Portal";
          case it:
            return (g.displayName || "Context") + ".Provider";
          case el:
            return (g._context.displayName || "Context") + ".Consumer";
          case Nt:
            var j = g.render;
            return g = g.displayName, g || (g = j.displayName || j.name || "", g = g !== "" ? "ForwardRef(" + g + ")" : "ForwardRef"), g;
          case De:
            return j = g.displayName || null, j !== null ? j : U(g.type) || "Memo";
          case bt:
            j = g._payload, g = g._init;
            try {
              return U(g(j));
            } catch {
            }
        }
      return null;
    }
    function W(g) {
      return "" + g;
    }
    function ge(g) {
      try {
        W(g);
        var j = !1;
      } catch {
        j = !0;
      }
      if (j) {
        j = console;
        var J = j.error, I = typeof Symbol == "function" && Symbol.toStringTag && g[Symbol.toStringTag] || g.constructor.name || "Object";
        return J.call(
          j,
          "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
          I
        ), W(g);
      }
    }
    function _(g) {
      if (g === At) return "<>";
      if (typeof g == "object" && g !== null && g.$$typeof === bt)
        return "<...>";
      try {
        var j = U(g);
        return j ? "<" + j + ">" : "<...>";
      } catch {
        return "<...>";
      }
    }
    function he() {
      var g = Se.A;
      return g === null ? null : g.getOwner();
    }
    function Ee() {
      return Error("react-stack-top-frame");
    }
    function Qe(g) {
      if (ze.call(g, "key")) {
        var j = Object.getOwnPropertyDescriptor(g, "key").get;
        if (j && j.isReactWarning) return !1;
      }
      return g.key !== void 0;
    }
    function et(g, j) {
      function J() {
        pt || (pt = !0, console.error(
          "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
          j
        ));
      }
      J.isReactWarning = !0, Object.defineProperty(g, "key", {
        get: J,
        configurable: !0
      });
    }
    function x() {
      var g = U(this.type);
      return O[g] || (O[g] = !0, console.error(
        "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
      )), g = this.props.ref, g !== void 0 ? g : null;
    }
    function w(g, j, J, I, ce, Oe, oe, il) {
      return J = Oe.ref, g = {
        $$typeof: ae,
        type: g,
        key: j,
        props: Oe,
        _owner: ce
      }, (J !== void 0 ? J : null) !== null ? Object.defineProperty(g, "ref", {
        enumerable: !1,
        get: x
      }) : Object.defineProperty(g, "ref", { enumerable: !1, value: null }), g._store = {}, Object.defineProperty(g._store, "validated", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: 0
      }), Object.defineProperty(g, "_debugInfo", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: null
      }), Object.defineProperty(g, "_debugStack", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: oe
      }), Object.defineProperty(g, "_debugTask", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: il
      }), Object.freeze && (Object.freeze(g.props), Object.freeze(g)), g;
    }
    function te(g, j, J, I, ce, Oe, oe, il) {
      var He = j.children;
      if (He !== void 0)
        if (I)
          if (Ot(He)) {
            for (I = 0; I < He.length; I++)
              G(He[I]);
            Object.freeze && Object.freeze(He);
          } else
            console.error(
              "React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead."
            );
        else G(He);
      if (ze.call(j, "key")) {
        He = U(g);
        var wt = Object.keys(j).filter(function(Dn) {
          return Dn !== "key";
        });
        I = 0 < wt.length ? "{key: someKey, " + wt.join(": ..., ") + ": ...}" : "{key: someKey}", be[He + I] || (wt = 0 < wt.length ? "{" + wt.join(": ..., ") + ": ...}" : "{}", console.error(
          `A props object containing a "key" prop is being spread into JSX:
  let props = %s;
  <%s {...props} />
React keys must be passed directly to JSX without using spread:
  let props = %s;
  <%s key={someKey} {...props} />`,
          I,
          He,
          wt,
          He
        ), be[He + I] = !0);
      }
      if (He = null, J !== void 0 && (ge(J), He = "" + J), Qe(j) && (ge(j.key), He = "" + j.key), "key" in j) {
        J = {};
        for (var na in j)
          na !== "key" && (J[na] = j[na]);
      } else J = j;
      return He && et(
        J,
        typeof g == "function" ? g.displayName || g.name || "Unknown" : g
      ), w(
        g,
        He,
        Oe,
        ce,
        he(),
        J,
        oe,
        il
      );
    }
    function G(g) {
      typeof g == "object" && g !== null && g.$$typeof === ae && g._store && (g._store.validated = 1);
    }
    var z = Ch(), ae = Symbol.for("react.transitional.element"), je = Symbol.for("react.portal"), At = Symbol.for("react.fragment"), ke = Symbol.for("react.strict_mode"), nt = Symbol.for("react.profiler"), el = Symbol.for("react.consumer"), it = Symbol.for("react.context"), Nt = Symbol.for("react.forward_ref"), se = Symbol.for("react.suspense"), Ue = Symbol.for("react.suspense_list"), De = Symbol.for("react.memo"), bt = Symbol.for("react.lazy"), re = Symbol.for("react.activity"), Rt = Symbol.for("react.client.reference"), Se = z.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, ze = Object.prototype.hasOwnProperty, Ot = Array.isArray, Gt = console.createTask ? console.createTask : function() {
      return null;
    };
    z = {
      react_stack_bottom_frame: function(g) {
        return g();
      }
    };
    var pt, O = {}, F = z.react_stack_bottom_frame.bind(
      z,
      Ee
    )(), P = Gt(_(Ee)), be = {};
    Ep.Fragment = At, Ep.jsx = function(g, j, J, I, ce) {
      var Oe = 1e4 > Se.recentlyCreatedOwnerStacks++;
      return te(
        g,
        j,
        J,
        !1,
        I,
        ce,
        Oe ? Error("react-stack-top-frame") : F,
        Oe ? Gt(_(g)) : P
      );
    }, Ep.jsxs = function(g, j, J, I, ce) {
      var Oe = 1e4 > Se.recentlyCreatedOwnerStacks++;
      return te(
        g,
        j,
        J,
        !0,
        I,
        ce,
        Oe ? Error("react-stack-top-frame") : F,
        Oe ? Gt(_(g)) : P
      );
    };
  }()), Ep;
}
var Kb;
function TT() {
  return Kb || (Kb = 1, It.env.NODE_ENV === "production" ? pg.exports = vT() : pg.exports = ST()), pg.exports;
}
var ie = TT(), gg = { exports: {} }, Ap = {}, bg = { exports: {} }, Y0 = {};
/**
 * @license React
 * scheduler.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var Jb;
function ET() {
  return Jb || (Jb = 1, function(U) {
    function W(O, F) {
      var P = O.length;
      O.push(F);
      e: for (; 0 < P; ) {
        var be = P - 1 >>> 1, g = O[be];
        if (0 < he(g, F))
          O[be] = F, O[P] = g, P = be;
        else break e;
      }
    }
    function ge(O) {
      return O.length === 0 ? null : O[0];
    }
    function _(O) {
      if (O.length === 0) return null;
      var F = O[0], P = O.pop();
      if (P !== F) {
        O[0] = P;
        e: for (var be = 0, g = O.length, j = g >>> 1; be < j; ) {
          var J = 2 * (be + 1) - 1, I = O[J], ce = J + 1, Oe = O[ce];
          if (0 > he(I, P))
            ce < g && 0 > he(Oe, I) ? (O[be] = Oe, O[ce] = P, be = ce) : (O[be] = I, O[J] = P, be = J);
          else if (ce < g && 0 > he(Oe, P))
            O[be] = Oe, O[ce] = P, be = ce;
          else break e;
        }
      }
      return F;
    }
    function he(O, F) {
      var P = O.sortIndex - F.sortIndex;
      return P !== 0 ? P : O.id - F.id;
    }
    if (U.unstable_now = void 0, typeof performance == "object" && typeof performance.now == "function") {
      var Ee = performance;
      U.unstable_now = function() {
        return Ee.now();
      };
    } else {
      var Qe = Date, et = Qe.now();
      U.unstable_now = function() {
        return Qe.now() - et;
      };
    }
    var x = [], w = [], te = 1, G = null, z = 3, ae = !1, je = !1, At = !1, ke = !1, nt = typeof setTimeout == "function" ? setTimeout : null, el = typeof clearTimeout == "function" ? clearTimeout : null, it = typeof setImmediate < "u" ? setImmediate : null;
    function Nt(O) {
      for (var F = ge(w); F !== null; ) {
        if (F.callback === null) _(w);
        else if (F.startTime <= O)
          _(w), F.sortIndex = F.expirationTime, W(x, F);
        else break;
        F = ge(w);
      }
    }
    function se(O) {
      if (At = !1, Nt(O), !je)
        if (ge(x) !== null)
          je = !0, Ue || (Ue = !0, ze());
        else {
          var F = ge(w);
          F !== null && pt(se, F.startTime - O);
        }
    }
    var Ue = !1, De = -1, bt = 5, re = -1;
    function Rt() {
      return ke ? !0 : !(U.unstable_now() - re < bt);
    }
    function Se() {
      if (ke = !1, Ue) {
        var O = U.unstable_now();
        re = O;
        var F = !0;
        try {
          e: {
            je = !1, At && (At = !1, el(De), De = -1), ae = !0;
            var P = z;
            try {
              t: {
                for (Nt(O), G = ge(x); G !== null && !(G.expirationTime > O && Rt()); ) {
                  var be = G.callback;
                  if (typeof be == "function") {
                    G.callback = null, z = G.priorityLevel;
                    var g = be(
                      G.expirationTime <= O
                    );
                    if (O = U.unstable_now(), typeof g == "function") {
                      G.callback = g, Nt(O), F = !0;
                      break t;
                    }
                    G === ge(x) && _(x), Nt(O);
                  } else _(x);
                  G = ge(x);
                }
                if (G !== null) F = !0;
                else {
                  var j = ge(w);
                  j !== null && pt(
                    se,
                    j.startTime - O
                  ), F = !1;
                }
              }
              break e;
            } finally {
              G = null, z = P, ae = !1;
            }
            F = void 0;
          }
        } finally {
          F ? ze() : Ue = !1;
        }
      }
    }
    var ze;
    if (typeof it == "function")
      ze = function() {
        it(Se);
      };
    else if (typeof MessageChannel < "u") {
      var Ot = new MessageChannel(), Gt = Ot.port2;
      Ot.port1.onmessage = Se, ze = function() {
        Gt.postMessage(null);
      };
    } else
      ze = function() {
        nt(Se, 0);
      };
    function pt(O, F) {
      De = nt(function() {
        O(U.unstable_now());
      }, F);
    }
    U.unstable_IdlePriority = 5, U.unstable_ImmediatePriority = 1, U.unstable_LowPriority = 4, U.unstable_NormalPriority = 3, U.unstable_Profiling = null, U.unstable_UserBlockingPriority = 2, U.unstable_cancelCallback = function(O) {
      O.callback = null;
    }, U.unstable_forceFrameRate = function(O) {
      0 > O || 125 < O ? console.error(
        "forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"
      ) : bt = 0 < O ? Math.floor(1e3 / O) : 5;
    }, U.unstable_getCurrentPriorityLevel = function() {
      return z;
    }, U.unstable_next = function(O) {
      switch (z) {
        case 1:
        case 2:
        case 3:
          var F = 3;
          break;
        default:
          F = z;
      }
      var P = z;
      z = F;
      try {
        return O();
      } finally {
        z = P;
      }
    }, U.unstable_requestPaint = function() {
      ke = !0;
    }, U.unstable_runWithPriority = function(O, F) {
      switch (O) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
          break;
        default:
          O = 3;
      }
      var P = z;
      z = O;
      try {
        return F();
      } finally {
        z = P;
      }
    }, U.unstable_scheduleCallback = function(O, F, P) {
      var be = U.unstable_now();
      switch (typeof P == "object" && P !== null ? (P = P.delay, P = typeof P == "number" && 0 < P ? be + P : be) : P = be, O) {
        case 1:
          var g = -1;
          break;
        case 2:
          g = 250;
          break;
        case 5:
          g = 1073741823;
          break;
        case 4:
          g = 1e4;
          break;
        default:
          g = 5e3;
      }
      return g = P + g, O = {
        id: te++,
        callback: F,
        priorityLevel: O,
        startTime: P,
        expirationTime: g,
        sortIndex: -1
      }, P > be ? (O.sortIndex = P, W(w, O), ge(x) === null && O === ge(w) && (At ? (el(De), De = -1) : At = !0, pt(se, P - be))) : (O.sortIndex = g, W(x, O), je || ae || (je = !0, Ue || (Ue = !0, ze()))), O;
    }, U.unstable_shouldYield = Rt, U.unstable_wrapCallback = function(O) {
      var F = z;
      return function() {
        var P = z;
        z = F;
        try {
          return O.apply(this, arguments);
        } finally {
          z = P;
        }
      };
    };
  }(Y0)), Y0;
}
var G0 = {}, kb;
function AT() {
  return kb || (kb = 1, function(U) {
    It.env.NODE_ENV !== "production" && function() {
      function W() {
        if (se = !1, re) {
          var O = U.unstable_now();
          ze = O;
          var F = !0;
          try {
            e: {
              it = !1, Nt && (Nt = !1, De(Rt), Rt = -1), el = !0;
              var P = nt;
              try {
                t: {
                  for (Qe(O), ke = _(ae); ke !== null && !(ke.expirationTime > O && x()); ) {
                    var be = ke.callback;
                    if (typeof be == "function") {
                      ke.callback = null, nt = ke.priorityLevel;
                      var g = be(
                        ke.expirationTime <= O
                      );
                      if (O = U.unstable_now(), typeof g == "function") {
                        ke.callback = g, Qe(O), F = !0;
                        break t;
                      }
                      ke === _(ae) && he(ae), Qe(O);
                    } else he(ae);
                    ke = _(ae);
                  }
                  if (ke !== null) F = !0;
                  else {
                    var j = _(je);
                    j !== null && w(
                      et,
                      j.startTime - O
                    ), F = !1;
                  }
                }
                break e;
              } finally {
                ke = null, nt = P, el = !1;
              }
              F = void 0;
            }
          } finally {
            F ? Ot() : re = !1;
          }
        }
      }
      function ge(O, F) {
        var P = O.length;
        O.push(F);
        e: for (; 0 < P; ) {
          var be = P - 1 >>> 1, g = O[be];
          if (0 < Ee(g, F))
            O[be] = F, O[P] = g, P = be;
          else break e;
        }
      }
      function _(O) {
        return O.length === 0 ? null : O[0];
      }
      function he(O) {
        if (O.length === 0) return null;
        var F = O[0], P = O.pop();
        if (P !== F) {
          O[0] = P;
          e: for (var be = 0, g = O.length, j = g >>> 1; be < j; ) {
            var J = 2 * (be + 1) - 1, I = O[J], ce = J + 1, Oe = O[ce];
            if (0 > Ee(I, P))
              ce < g && 0 > Ee(Oe, I) ? (O[be] = Oe, O[ce] = P, be = ce) : (O[be] = I, O[J] = P, be = J);
            else if (ce < g && 0 > Ee(Oe, P))
              O[be] = Oe, O[ce] = P, be = ce;
            else break e;
          }
        }
        return F;
      }
      function Ee(O, F) {
        var P = O.sortIndex - F.sortIndex;
        return P !== 0 ? P : O.id - F.id;
      }
      function Qe(O) {
        for (var F = _(je); F !== null; ) {
          if (F.callback === null) he(je);
          else if (F.startTime <= O)
            he(je), F.sortIndex = F.expirationTime, ge(ae, F);
          else break;
          F = _(je);
        }
      }
      function et(O) {
        if (Nt = !1, Qe(O), !it)
          if (_(ae) !== null)
            it = !0, re || (re = !0, Ot());
          else {
            var F = _(je);
            F !== null && w(
              et,
              F.startTime - O
            );
          }
      }
      function x() {
        return se ? !0 : !(U.unstable_now() - ze < Se);
      }
      function w(O, F) {
        Rt = Ue(function() {
          O(U.unstable_now());
        }, F);
      }
      if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error()), U.unstable_now = void 0, typeof performance == "object" && typeof performance.now == "function") {
        var te = performance;
        U.unstable_now = function() {
          return te.now();
        };
      } else {
        var G = Date, z = G.now();
        U.unstable_now = function() {
          return G.now() - z;
        };
      }
      var ae = [], je = [], At = 1, ke = null, nt = 3, el = !1, it = !1, Nt = !1, se = !1, Ue = typeof setTimeout == "function" ? setTimeout : null, De = typeof clearTimeout == "function" ? clearTimeout : null, bt = typeof setImmediate < "u" ? setImmediate : null, re = !1, Rt = -1, Se = 5, ze = -1;
      if (typeof bt == "function")
        var Ot = function() {
          bt(W);
        };
      else if (typeof MessageChannel < "u") {
        var Gt = new MessageChannel(), pt = Gt.port2;
        Gt.port1.onmessage = W, Ot = function() {
          pt.postMessage(null);
        };
      } else
        Ot = function() {
          Ue(W, 0);
        };
      U.unstable_IdlePriority = 5, U.unstable_ImmediatePriority = 1, U.unstable_LowPriority = 4, U.unstable_NormalPriority = 3, U.unstable_Profiling = null, U.unstable_UserBlockingPriority = 2, U.unstable_cancelCallback = function(O) {
        O.callback = null;
      }, U.unstable_forceFrameRate = function(O) {
        0 > O || 125 < O ? console.error(
          "forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"
        ) : Se = 0 < O ? Math.floor(1e3 / O) : 5;
      }, U.unstable_getCurrentPriorityLevel = function() {
        return nt;
      }, U.unstable_next = function(O) {
        switch (nt) {
          case 1:
          case 2:
          case 3:
            var F = 3;
            break;
          default:
            F = nt;
        }
        var P = nt;
        nt = F;
        try {
          return O();
        } finally {
          nt = P;
        }
      }, U.unstable_requestPaint = function() {
        se = !0;
      }, U.unstable_runWithPriority = function(O, F) {
        switch (O) {
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
            break;
          default:
            O = 3;
        }
        var P = nt;
        nt = O;
        try {
          return F();
        } finally {
          nt = P;
        }
      }, U.unstable_scheduleCallback = function(O, F, P) {
        var be = U.unstable_now();
        switch (typeof P == "object" && P !== null ? (P = P.delay, P = typeof P == "number" && 0 < P ? be + P : be) : P = be, O) {
          case 1:
            var g = -1;
            break;
          case 2:
            g = 250;
            break;
          case 5:
            g = 1073741823;
            break;
          case 4:
            g = 1e4;
            break;
          default:
            g = 5e3;
        }
        return g = P + g, O = {
          id: At++,
          callback: F,
          priorityLevel: O,
          startTime: P,
          expirationTime: g,
          sortIndex: -1
        }, P > be ? (O.sortIndex = P, ge(je, O), _(ae) === null && O === _(je) && (Nt ? (De(Rt), Rt = -1) : Nt = !0, w(et, P - be))) : (O.sortIndex = g, ge(ae, O), it || el || (it = !0, re || (re = !0, Ot()))), O;
      }, U.unstable_shouldYield = x, U.unstable_wrapCallback = function(O) {
        var F = nt;
        return function() {
          var P = nt;
          nt = F;
          try {
            return O.apply(this, arguments);
          } finally {
            nt = P;
          }
        };
      }, typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    }();
  }(G0)), G0;
}
var $b;
function iS() {
  return $b || ($b = 1, It.env.NODE_ENV === "production" ? bg.exports = ET() : bg.exports = AT()), bg.exports;
}
var Sg = { exports: {} }, Ta = {};
/**
 * @license React
 * react-dom.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var Wb;
function RT() {
  if (Wb) return Ta;
  Wb = 1;
  var U = Ch();
  function W(x) {
    var w = "https://react.dev/errors/" + x;
    if (1 < arguments.length) {
      w += "?args[]=" + encodeURIComponent(arguments[1]);
      for (var te = 2; te < arguments.length; te++)
        w += "&args[]=" + encodeURIComponent(arguments[te]);
    }
    return "Minified React error #" + x + "; visit " + w + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  function ge() {
  }
  var _ = {
    d: {
      f: ge,
      r: function() {
        throw Error(W(522));
      },
      D: ge,
      C: ge,
      L: ge,
      m: ge,
      X: ge,
      S: ge,
      M: ge
    },
    p: 0,
    findDOMNode: null
  }, he = Symbol.for("react.portal");
  function Ee(x, w, te) {
    var G = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: he,
      key: G == null ? null : "" + G,
      children: x,
      containerInfo: w,
      implementation: te
    };
  }
  var Qe = U.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  function et(x, w) {
    if (x === "font") return "";
    if (typeof w == "string")
      return w === "use-credentials" ? w : "";
  }
  return Ta.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = _, Ta.createPortal = function(x, w) {
    var te = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!w || w.nodeType !== 1 && w.nodeType !== 9 && w.nodeType !== 11)
      throw Error(W(299));
    return Ee(x, w, null, te);
  }, Ta.flushSync = function(x) {
    var w = Qe.T, te = _.p;
    try {
      if (Qe.T = null, _.p = 2, x) return x();
    } finally {
      Qe.T = w, _.p = te, _.d.f();
    }
  }, Ta.preconnect = function(x, w) {
    typeof x == "string" && (w ? (w = w.crossOrigin, w = typeof w == "string" ? w === "use-credentials" ? w : "" : void 0) : w = null, _.d.C(x, w));
  }, Ta.prefetchDNS = function(x) {
    typeof x == "string" && _.d.D(x);
  }, Ta.preinit = function(x, w) {
    if (typeof x == "string" && w && typeof w.as == "string") {
      var te = w.as, G = et(te, w.crossOrigin), z = typeof w.integrity == "string" ? w.integrity : void 0, ae = typeof w.fetchPriority == "string" ? w.fetchPriority : void 0;
      te === "style" ? _.d.S(
        x,
        typeof w.precedence == "string" ? w.precedence : void 0,
        {
          crossOrigin: G,
          integrity: z,
          fetchPriority: ae
        }
      ) : te === "script" && _.d.X(x, {
        crossOrigin: G,
        integrity: z,
        fetchPriority: ae,
        nonce: typeof w.nonce == "string" ? w.nonce : void 0
      });
    }
  }, Ta.preinitModule = function(x, w) {
    if (typeof x == "string")
      if (typeof w == "object" && w !== null) {
        if (w.as == null || w.as === "script") {
          var te = et(
            w.as,
            w.crossOrigin
          );
          _.d.M(x, {
            crossOrigin: te,
            integrity: typeof w.integrity == "string" ? w.integrity : void 0,
            nonce: typeof w.nonce == "string" ? w.nonce : void 0
          });
        }
      } else w == null && _.d.M(x);
  }, Ta.preload = function(x, w) {
    if (typeof x == "string" && typeof w == "object" && w !== null && typeof w.as == "string") {
      var te = w.as, G = et(te, w.crossOrigin);
      _.d.L(x, te, {
        crossOrigin: G,
        integrity: typeof w.integrity == "string" ? w.integrity : void 0,
        nonce: typeof w.nonce == "string" ? w.nonce : void 0,
        type: typeof w.type == "string" ? w.type : void 0,
        fetchPriority: typeof w.fetchPriority == "string" ? w.fetchPriority : void 0,
        referrerPolicy: typeof w.referrerPolicy == "string" ? w.referrerPolicy : void 0,
        imageSrcSet: typeof w.imageSrcSet == "string" ? w.imageSrcSet : void 0,
        imageSizes: typeof w.imageSizes == "string" ? w.imageSizes : void 0,
        media: typeof w.media == "string" ? w.media : void 0
      });
    }
  }, Ta.preloadModule = function(x, w) {
    if (typeof x == "string")
      if (w) {
        var te = et(w.as, w.crossOrigin);
        _.d.m(x, {
          as: typeof w.as == "string" && w.as !== "script" ? w.as : void 0,
          crossOrigin: te,
          integrity: typeof w.integrity == "string" ? w.integrity : void 0
        });
      } else _.d.m(x);
  }, Ta.requestFormReset = function(x) {
    _.d.r(x);
  }, Ta.unstable_batchedUpdates = function(x, w) {
    return x(w);
  }, Ta.useFormState = function(x, w, te) {
    return Qe.H.useFormState(x, w, te);
  }, Ta.useFormStatus = function() {
    return Qe.H.useHostTransitionStatus();
  }, Ta.version = "19.1.1", Ta;
}
var Ea = {}, Fb;
function OT() {
  return Fb || (Fb = 1, It.env.NODE_ENV !== "production" && function() {
    function U() {
    }
    function W(G) {
      return "" + G;
    }
    function ge(G, z, ae) {
      var je = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
      try {
        W(je);
        var At = !1;
      } catch {
        At = !0;
      }
      return At && (console.error(
        "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
        typeof Symbol == "function" && Symbol.toStringTag && je[Symbol.toStringTag] || je.constructor.name || "Object"
      ), W(je)), {
        $$typeof: w,
        key: je == null ? null : "" + je,
        children: G,
        containerInfo: z,
        implementation: ae
      };
    }
    function _(G, z) {
      if (G === "font") return "";
      if (typeof z == "string")
        return z === "use-credentials" ? z : "";
    }
    function he(G) {
      return G === null ? "`null`" : G === void 0 ? "`undefined`" : G === "" ? "an empty string" : 'something with type "' + typeof G + '"';
    }
    function Ee(G) {
      return G === null ? "`null`" : G === void 0 ? "`undefined`" : G === "" ? "an empty string" : typeof G == "string" ? JSON.stringify(G) : typeof G == "number" ? "`" + G + "`" : 'something with type "' + typeof G + '"';
    }
    function Qe() {
      var G = te.H;
      return G === null && console.error(
        `Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:
1. You might have mismatching versions of React and the renderer (such as React DOM)
2. You might be breaking the Rules of Hooks
3. You might have more than one copy of React in the same app
See https://react.dev/link/invalid-hook-call for tips about how to debug and fix this problem.`
      ), G;
    }
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var et = Ch(), x = {
      d: {
        f: U,
        r: function() {
          throw Error(
            "Invalid form element. requestFormReset must be passed a form that was rendered by React."
          );
        },
        D: U,
        C: U,
        L: U,
        m: U,
        X: U,
        S: U,
        M: U
      },
      p: 0,
      findDOMNode: null
    }, w = Symbol.for("react.portal"), te = et.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    typeof Map == "function" && Map.prototype != null && typeof Map.prototype.forEach == "function" && typeof Set == "function" && Set.prototype != null && typeof Set.prototype.clear == "function" && typeof Set.prototype.forEach == "function" || console.error(
      "React depends on Map and Set built-in types. Make sure that you load a polyfill in older browsers. https://reactjs.org/link/react-polyfills"
    ), Ea.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = x, Ea.createPortal = function(G, z) {
      var ae = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
      if (!z || z.nodeType !== 1 && z.nodeType !== 9 && z.nodeType !== 11)
        throw Error("Target container is not a DOM element.");
      return ge(G, z, null, ae);
    }, Ea.flushSync = function(G) {
      var z = te.T, ae = x.p;
      try {
        if (te.T = null, x.p = 2, G)
          return G();
      } finally {
        te.T = z, x.p = ae, x.d.f() && console.error(
          "flushSync was called from inside a lifecycle method. React cannot flush when React is already rendering. Consider moving this call to a scheduler task or micro task."
        );
      }
    }, Ea.preconnect = function(G, z) {
      typeof G == "string" && G ? z != null && typeof z != "object" ? console.error(
        "ReactDOM.preconnect(): Expected the `options` argument (second) to be an object but encountered %s instead. The only supported option at this time is `crossOrigin` which accepts a string.",
        Ee(z)
      ) : z != null && typeof z.crossOrigin != "string" && console.error(
        "ReactDOM.preconnect(): Expected the `crossOrigin` option (second argument) to be a string but encountered %s instead. Try removing this option or passing a string value instead.",
        he(z.crossOrigin)
      ) : console.error(
        "ReactDOM.preconnect(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
        he(G)
      ), typeof G == "string" && (z ? (z = z.crossOrigin, z = typeof z == "string" ? z === "use-credentials" ? z : "" : void 0) : z = null, x.d.C(G, z));
    }, Ea.prefetchDNS = function(G) {
      if (typeof G != "string" || !G)
        console.error(
          "ReactDOM.prefetchDNS(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
          he(G)
        );
      else if (1 < arguments.length) {
        var z = arguments[1];
        typeof z == "object" && z.hasOwnProperty("crossOrigin") ? console.error(
          "ReactDOM.prefetchDNS(): Expected only one argument, `href`, but encountered %s as a second argument instead. This argument is reserved for future options and is currently disallowed. It looks like the you are attempting to set a crossOrigin property for this DNS lookup hint. Browsers do not perform DNS queries using CORS and setting this attribute on the resource hint has no effect. Try calling ReactDOM.prefetchDNS() with just a single string argument, `href`.",
          Ee(z)
        ) : console.error(
          "ReactDOM.prefetchDNS(): Expected only one argument, `href`, but encountered %s as a second argument instead. This argument is reserved for future options and is currently disallowed. Try calling ReactDOM.prefetchDNS() with just a single string argument, `href`.",
          Ee(z)
        );
      }
      typeof G == "string" && x.d.D(G);
    }, Ea.preinit = function(G, z) {
      if (typeof G == "string" && G ? z == null || typeof z != "object" ? console.error(
        "ReactDOM.preinit(): Expected the `options` argument (second) to be an object with an `as` property describing the type of resource to be preinitialized but encountered %s instead.",
        Ee(z)
      ) : z.as !== "style" && z.as !== "script" && console.error(
        'ReactDOM.preinit(): Expected the `as` property in the `options` argument (second) to contain a valid value describing the type of resource to be preinitialized but encountered %s instead. Valid values for `as` are "style" and "script".',
        Ee(z.as)
      ) : console.error(
        "ReactDOM.preinit(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
        he(G)
      ), typeof G == "string" && z && typeof z.as == "string") {
        var ae = z.as, je = _(ae, z.crossOrigin), At = typeof z.integrity == "string" ? z.integrity : void 0, ke = typeof z.fetchPriority == "string" ? z.fetchPriority : void 0;
        ae === "style" ? x.d.S(
          G,
          typeof z.precedence == "string" ? z.precedence : void 0,
          {
            crossOrigin: je,
            integrity: At,
            fetchPriority: ke
          }
        ) : ae === "script" && x.d.X(G, {
          crossOrigin: je,
          integrity: At,
          fetchPriority: ke,
          nonce: typeof z.nonce == "string" ? z.nonce : void 0
        });
      }
    }, Ea.preinitModule = function(G, z) {
      var ae = "";
      if (typeof G == "string" && G || (ae += " The `href` argument encountered was " + he(G) + "."), z !== void 0 && typeof z != "object" ? ae += " The `options` argument encountered was " + he(z) + "." : z && "as" in z && z.as !== "script" && (ae += " The `as` option encountered was " + Ee(z.as) + "."), ae)
        console.error(
          "ReactDOM.preinitModule(): Expected up to two arguments, a non-empty `href` string and, optionally, an `options` object with a valid `as` property.%s",
          ae
        );
      else
        switch (ae = z && typeof z.as == "string" ? z.as : "script", ae) {
          case "script":
            break;
          default:
            ae = Ee(ae), console.error(
              'ReactDOM.preinitModule(): Currently the only supported "as" type for this function is "script" but received "%s" instead. This warning was generated for `href` "%s". In the future other module types will be supported, aligning with the import-attributes proposal. Learn more here: (https://github.com/tc39/proposal-import-attributes)',
              ae,
              G
            );
        }
      typeof G == "string" && (typeof z == "object" && z !== null ? (z.as == null || z.as === "script") && (ae = _(
        z.as,
        z.crossOrigin
      ), x.d.M(G, {
        crossOrigin: ae,
        integrity: typeof z.integrity == "string" ? z.integrity : void 0,
        nonce: typeof z.nonce == "string" ? z.nonce : void 0
      })) : z == null && x.d.M(G));
    }, Ea.preload = function(G, z) {
      var ae = "";
      if (typeof G == "string" && G || (ae += " The `href` argument encountered was " + he(G) + "."), z == null || typeof z != "object" ? ae += " The `options` argument encountered was " + he(z) + "." : typeof z.as == "string" && z.as || (ae += " The `as` option encountered was " + he(z.as) + "."), ae && console.error(
        'ReactDOM.preload(): Expected two arguments, a non-empty `href` string and an `options` object with an `as` property valid for a `<link rel="preload" as="..." />` tag.%s',
        ae
      ), typeof G == "string" && typeof z == "object" && z !== null && typeof z.as == "string") {
        ae = z.as;
        var je = _(
          ae,
          z.crossOrigin
        );
        x.d.L(G, ae, {
          crossOrigin: je,
          integrity: typeof z.integrity == "string" ? z.integrity : void 0,
          nonce: typeof z.nonce == "string" ? z.nonce : void 0,
          type: typeof z.type == "string" ? z.type : void 0,
          fetchPriority: typeof z.fetchPriority == "string" ? z.fetchPriority : void 0,
          referrerPolicy: typeof z.referrerPolicy == "string" ? z.referrerPolicy : void 0,
          imageSrcSet: typeof z.imageSrcSet == "string" ? z.imageSrcSet : void 0,
          imageSizes: typeof z.imageSizes == "string" ? z.imageSizes : void 0,
          media: typeof z.media == "string" ? z.media : void 0
        });
      }
    }, Ea.preloadModule = function(G, z) {
      var ae = "";
      typeof G == "string" && G || (ae += " The `href` argument encountered was " + he(G) + "."), z !== void 0 && typeof z != "object" ? ae += " The `options` argument encountered was " + he(z) + "." : z && "as" in z && typeof z.as != "string" && (ae += " The `as` option encountered was " + he(z.as) + "."), ae && console.error(
        'ReactDOM.preloadModule(): Expected two arguments, a non-empty `href` string and, optionally, an `options` object with an `as` property valid for a `<link rel="modulepreload" as="..." />` tag.%s',
        ae
      ), typeof G == "string" && (z ? (ae = _(
        z.as,
        z.crossOrigin
      ), x.d.m(G, {
        as: typeof z.as == "string" && z.as !== "script" ? z.as : void 0,
        crossOrigin: ae,
        integrity: typeof z.integrity == "string" ? z.integrity : void 0
      })) : x.d.m(G));
    }, Ea.requestFormReset = function(G) {
      x.d.r(G);
    }, Ea.unstable_batchedUpdates = function(G, z) {
      return G(z);
    }, Ea.useFormState = function(G, z, ae) {
      return Qe().useFormState(G, z, ae);
    }, Ea.useFormStatus = function() {
      return Qe().useHostTransitionStatus();
    }, Ea.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  }()), Ea;
}
var Ib;
function cS() {
  if (Ib) return Sg.exports;
  Ib = 1;
  function U() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) {
      if (It.env.NODE_ENV !== "production")
        throw new Error("^_^");
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(U);
      } catch (W) {
        console.error(W);
      }
    }
  }
  return It.env.NODE_ENV === "production" ? (U(), Sg.exports = RT()) : Sg.exports = OT(), Sg.exports;
}
var Pb;
function DT() {
  if (Pb) return Ap;
  Pb = 1;
  var U = iS(), W = Ch(), ge = cS();
  function _(l) {
    var n = "https://react.dev/errors/" + l;
    if (1 < arguments.length) {
      n += "?args[]=" + encodeURIComponent(arguments[1]);
      for (var u = 2; u < arguments.length; u++)
        n += "&args[]=" + encodeURIComponent(arguments[u]);
    }
    return "Minified React error #" + l + "; visit " + n + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  function he(l) {
    return !(!l || l.nodeType !== 1 && l.nodeType !== 9 && l.nodeType !== 11);
  }
  function Ee(l) {
    var n = l, u = l;
    if (l.alternate) for (; n.return; ) n = n.return;
    else {
      l = n;
      do
        n = l, (n.flags & 4098) !== 0 && (u = n.return), l = n.return;
      while (l);
    }
    return n.tag === 3 ? u : null;
  }
  function Qe(l) {
    if (l.tag === 13) {
      var n = l.memoizedState;
      if (n === null && (l = l.alternate, l !== null && (n = l.memoizedState)), n !== null) return n.dehydrated;
    }
    return null;
  }
  function et(l) {
    if (Ee(l) !== l)
      throw Error(_(188));
  }
  function x(l) {
    var n = l.alternate;
    if (!n) {
      if (n = Ee(l), n === null) throw Error(_(188));
      return n !== l ? null : l;
    }
    for (var u = l, c = n; ; ) {
      var s = u.return;
      if (s === null) break;
      var r = s.alternate;
      if (r === null) {
        if (c = s.return, c !== null) {
          u = c;
          continue;
        }
        break;
      }
      if (s.child === r.child) {
        for (r = s.child; r; ) {
          if (r === u) return et(s), l;
          if (r === c) return et(s), n;
          r = r.sibling;
        }
        throw Error(_(188));
      }
      if (u.return !== c.return) u = s, c = r;
      else {
        for (var y = !1, p = s.child; p; ) {
          if (p === u) {
            y = !0, u = s, c = r;
            break;
          }
          if (p === c) {
            y = !0, c = s, u = r;
            break;
          }
          p = p.sibling;
        }
        if (!y) {
          for (p = r.child; p; ) {
            if (p === u) {
              y = !0, u = r, c = s;
              break;
            }
            if (p === c) {
              y = !0, c = r, u = s;
              break;
            }
            p = p.sibling;
          }
          if (!y) throw Error(_(189));
        }
      }
      if (u.alternate !== c) throw Error(_(190));
    }
    if (u.tag !== 3) throw Error(_(188));
    return u.stateNode.current === u ? l : n;
  }
  function w(l) {
    var n = l.tag;
    if (n === 5 || n === 26 || n === 27 || n === 6) return l;
    for (l = l.child; l !== null; ) {
      if (n = w(l), n !== null) return n;
      l = l.sibling;
    }
    return null;
  }
  var te = Object.assign, G = Symbol.for("react.element"), z = Symbol.for("react.transitional.element"), ae = Symbol.for("react.portal"), je = Symbol.for("react.fragment"), At = Symbol.for("react.strict_mode"), ke = Symbol.for("react.profiler"), nt = Symbol.for("react.provider"), el = Symbol.for("react.consumer"), it = Symbol.for("react.context"), Nt = Symbol.for("react.forward_ref"), se = Symbol.for("react.suspense"), Ue = Symbol.for("react.suspense_list"), De = Symbol.for("react.memo"), bt = Symbol.for("react.lazy"), re = Symbol.for("react.activity"), Rt = Symbol.for("react.memo_cache_sentinel"), Se = Symbol.iterator;
  function ze(l) {
    return l === null || typeof l != "object" ? null : (l = Se && l[Se] || l["@@iterator"], typeof l == "function" ? l : null);
  }
  var Ot = Symbol.for("react.client.reference");
  function Gt(l) {
    if (l == null) return null;
    if (typeof l == "function")
      return l.$$typeof === Ot ? null : l.displayName || l.name || null;
    if (typeof l == "string") return l;
    switch (l) {
      case je:
        return "Fragment";
      case ke:
        return "Profiler";
      case At:
        return "StrictMode";
      case se:
        return "Suspense";
      case Ue:
        return "SuspenseList";
      case re:
        return "Activity";
    }
    if (typeof l == "object")
      switch (l.$$typeof) {
        case ae:
          return "Portal";
        case it:
          return (l.displayName || "Context") + ".Provider";
        case el:
          return (l._context.displayName || "Context") + ".Consumer";
        case Nt:
          var n = l.render;
          return l = l.displayName, l || (l = n.displayName || n.name || "", l = l !== "" ? "ForwardRef(" + l + ")" : "ForwardRef"), l;
        case De:
          return n = l.displayName || null, n !== null ? n : Gt(l.type) || "Memo";
        case bt:
          n = l._payload, l = l._init;
          try {
            return Gt(l(n));
          } catch {
          }
      }
    return null;
  }
  var pt = Array.isArray, O = W.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, F = ge.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, P = {
    pending: !1,
    data: null,
    method: null,
    action: null
  }, be = [], g = -1;
  function j(l) {
    return { current: l };
  }
  function J(l) {
    0 > g || (l.current = be[g], be[g] = null, g--);
  }
  function I(l, n) {
    g++, be[g] = l.current, l.current = n;
  }
  var ce = j(null), Oe = j(null), oe = j(null), il = j(null);
  function He(l, n) {
    switch (I(oe, n), I(Oe, l), I(ce, null), n.nodeType) {
      case 9:
      case 11:
        l = (l = n.documentElement) && (l = l.namespaceURI) ? Gu(l) : 0;
        break;
      default:
        if (l = n.tagName, n = n.namespaceURI)
          n = Gu(n), l = ko(n, l);
        else
          switch (l) {
            case "svg":
              l = 1;
              break;
            case "math":
              l = 2;
              break;
            default:
              l = 0;
          }
    }
    J(ce), I(ce, l);
  }
  function wt() {
    J(ce), J(Oe), J(oe);
  }
  function na(l) {
    l.memoizedState !== null && I(il, l);
    var n = ce.current, u = ko(n, l.type);
    n !== u && (I(Oe, l), I(ce, u));
  }
  function Dn(l) {
    Oe.current === l && (J(ce), J(Oe)), il.current === l && (J(il), ba._currentValue = P);
  }
  var Zi = Object.prototype.hasOwnProperty, zn = U.unstable_scheduleCallback, Pc = U.unstable_cancelCallback, bf = U.unstable_shouldYield, tl = U.unstable_requestPaint, pl = U.unstable_now, Iu = U.unstable_getCurrentPriorityLevel, ir = U.unstable_ImmediatePriority, Ke = U.unstable_UserBlockingPriority, Mn = U.unstable_NormalPriority, eo = U.unstable_LowPriority, gu = U.unstable_IdlePriority, cr = U.log, Sf = U.unstable_setDisableYieldValue, Pu = null, Ol = null;
  function Ba(l) {
    if (typeof cr == "function" && Sf(l), Ol && typeof Ol.setStrictMode == "function")
      try {
        Ol.setStrictMode(Pu, l);
      } catch {
      }
  }
  var Dl = Math.clz32 ? Math.clz32 : lo, to = Math.log, Tf = Math.LN2;
  function lo(l) {
    return l >>>= 0, l === 0 ? 32 : 31 - (to(l) / Tf | 0) | 0;
  }
  var nn = 256, ua = 4194304;
  function zl(l) {
    var n = l & 42;
    if (n !== 0) return n;
    switch (l & -l) {
      case 1:
        return 1;
      case 2:
        return 2;
      case 4:
        return 4;
      case 8:
        return 8;
      case 16:
        return 16;
      case 32:
        return 32;
      case 64:
        return 64;
      case 128:
        return 128;
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return l & 4194048;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
        return l & 62914560;
      case 67108864:
        return 67108864;
      case 134217728:
        return 134217728;
      case 268435456:
        return 268435456;
      case 536870912:
        return 536870912;
      case 1073741824:
        return 0;
      default:
        return l;
    }
  }
  function un(l, n, u) {
    var c = l.pendingLanes;
    if (c === 0) return 0;
    var s = 0, r = l.suspendedLanes, y = l.pingedLanes;
    l = l.warmLanes;
    var p = c & 134217727;
    return p !== 0 ? (c = p & ~r, c !== 0 ? s = zl(c) : (y &= p, y !== 0 ? s = zl(y) : u || (u = p & ~l, u !== 0 && (s = zl(u))))) : (p = c & ~r, p !== 0 ? s = zl(p) : y !== 0 ? s = zl(y) : u || (u = c & ~l, u !== 0 && (s = zl(u)))), s === 0 ? 0 : n !== 0 && n !== s && (n & r) === 0 && (r = s & -s, u = n & -n, r >= u || r === 32 && (u & 4194048) !== 0) ? n : s;
  }
  function m(l, n) {
    return (l.pendingLanes & ~(l.suspendedLanes & ~l.pingedLanes) & n) === 0;
  }
  function D(l, n) {
    switch (l) {
      case 1:
      case 2:
      case 4:
      case 8:
      case 64:
        return n + 250;
      case 16:
      case 32:
      case 128:
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return n + 5e3;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
        return -1;
      case 67108864:
      case 134217728:
      case 268435456:
      case 536870912:
      case 1073741824:
        return -1;
      default:
        return -1;
    }
  }
  function le() {
    var l = nn;
    return nn <<= 1, (nn & 4194048) === 0 && (nn = 256), l;
  }
  function ue() {
    var l = ua;
    return ua <<= 1, (ua & 62914560) === 0 && (ua = 4194304), l;
  }
  function ve(l) {
    for (var n = [], u = 0; 31 > u; u++) n.push(l);
    return n;
  }
  function Ne(l, n) {
    l.pendingLanes |= n, n !== 268435456 && (l.suspendedLanes = 0, l.pingedLanes = 0, l.warmLanes = 0);
  }
  function Ye(l, n, u, c, s, r) {
    var y = l.pendingLanes;
    l.pendingLanes = u, l.suspendedLanes = 0, l.pingedLanes = 0, l.warmLanes = 0, l.expiredLanes &= u, l.entangledLanes &= u, l.errorRecoveryDisabledLanes &= u, l.shellSuspendCounter = 0;
    var p = l.entanglements, S = l.expirationTimes, H = l.hiddenUpdates;
    for (u = y & ~u; 0 < u; ) {
      var K = 31 - Dl(u), $ = 1 << K;
      p[K] = 0, S[K] = -1;
      var q = H[K];
      if (q !== null)
        for (H[K] = null, K = 0; K < q.length; K++) {
          var Y = q[K];
          Y !== null && (Y.lane &= -536870913);
        }
      u &= ~$;
    }
    c !== 0 && ct(l, c, 0), r !== 0 && s === 0 && l.tag !== 0 && (l.suspendedLanes |= r & ~(y & ~n));
  }
  function ct(l, n, u) {
    l.pendingLanes |= n, l.suspendedLanes &= ~n;
    var c = 31 - Dl(n);
    l.entangledLanes |= n, l.entanglements[c] = l.entanglements[c] | 1073741824 | u & 4194090;
  }
  function Be(l, n) {
    var u = l.entangledLanes |= n;
    for (l = l.entanglements; u; ) {
      var c = 31 - Dl(u), s = 1 << c;
      s & n | l[c] & n && (l[c] |= n), u &= ~s;
    }
  }
  function ll(l) {
    switch (l) {
      case 2:
        l = 1;
        break;
      case 8:
        l = 4;
        break;
      case 32:
        l = 16;
        break;
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
        l = 128;
        break;
      case 268435456:
        l = 134217728;
        break;
      default:
        l = 0;
    }
    return l;
  }
  function cn(l) {
    return l &= -l, 2 < l ? 8 < l ? (l & 134217727) !== 0 ? 32 : 268435456 : 8 : 2;
  }
  function or() {
    var l = F.p;
    return l !== 0 ? l : (l = window.event, l === void 0 ? 32 : qm(l.type));
  }
  function xh(l, n) {
    var u = F.p;
    try {
      return F.p = l, n();
    } finally {
      F.p = u;
    }
  }
  var cl = Math.random().toString(36).slice(2), vl = "__reactFiber$" + cl, kl = "__reactProps$" + cl, ao = "__reactContainer$" + cl, fr = "__reactEvents$" + cl, Dp = "__reactListeners$" + cl, sr = "__reactHandles$" + cl, zp = "__reactResources$" + cl, ye = "__reactMarker$" + cl;
  function Ef(l) {
    delete l[vl], delete l[kl], delete l[fr], delete l[Dp], delete l[sr];
  }
  function Ml(l) {
    var n = l[vl];
    if (n) return n;
    for (var u = l.parentNode; u; ) {
      if (n = u[ao] || u[vl]) {
        if (u = n.alternate, n.child !== null || u !== null && u.child !== null)
          for (l = wl(l); l !== null; ) {
            if (u = l[vl]) return u;
            l = wl(l);
          }
        return n;
      }
      l = u, u = l.parentNode;
    }
    return null;
  }
  function Ki(l) {
    if (l = l[vl] || l[ao]) {
      var n = l.tag;
      if (n === 5 || n === 6 || n === 13 || n === 26 || n === 27 || n === 3)
        return l;
    }
    return null;
  }
  function Af(l) {
    var n = l.tag;
    if (n === 5 || n === 26 || n === 27 || n === 6) return l.stateNode;
    throw Error(_(33));
  }
  function bu(l) {
    var n = l[zp];
    return n || (n = l[zp] = { hoistableStyles: /* @__PURE__ */ new Map(), hoistableScripts: /* @__PURE__ */ new Map() }), n;
  }
  function ol(l) {
    l[ye] = !0;
  }
  var Rf = /* @__PURE__ */ new Set(), Aa = {};
  function ei(l, n) {
    ti(l, n), ti(l + "Capture", n);
  }
  function ti(l, n) {
    for (Aa[l] = n, l = 0; l < n.length; l++)
      Rf.add(n[l]);
  }
  var Mp = RegExp(
    "^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
  ), rr = {}, Hh = {};
  function _p(l) {
    return Zi.call(Hh, l) ? !0 : Zi.call(rr, l) ? !1 : Mp.test(l) ? Hh[l] = !0 : (rr[l] = !0, !1);
  }
  function Su(l, n, u) {
    if (_p(n))
      if (u === null) l.removeAttribute(n);
      else {
        switch (typeof u) {
          case "undefined":
          case "function":
          case "symbol":
            l.removeAttribute(n);
            return;
          case "boolean":
            var c = n.toLowerCase().slice(0, 5);
            if (c !== "data-" && c !== "aria-") {
              l.removeAttribute(n);
              return;
            }
        }
        l.setAttribute(n, "" + u);
      }
  }
  function Of(l, n, u) {
    if (u === null) l.removeAttribute(n);
    else {
      switch (typeof u) {
        case "undefined":
        case "function":
        case "symbol":
        case "boolean":
          l.removeAttribute(n);
          return;
      }
      l.setAttribute(n, "" + u);
    }
  }
  function _n(l, n, u, c) {
    if (c === null) l.removeAttribute(u);
    else {
      switch (typeof c) {
        case "undefined":
        case "function":
        case "symbol":
        case "boolean":
          l.removeAttribute(u);
          return;
      }
      l.setAttributeNS(n, u, "" + c);
    }
  }
  var dr, Nh;
  function Ji(l) {
    if (dr === void 0)
      try {
        throw Error();
      } catch (u) {
        var n = u.stack.trim().match(/\n( *(at )?)/);
        dr = n && n[1] || "", Nh = -1 < u.stack.indexOf(`
    at`) ? " (<anonymous>)" : -1 < u.stack.indexOf("@") ? "@unknown:0:0" : "";
      }
    return `
` + dr + l + Nh;
  }
  var $l = !1;
  function li(l, n) {
    if (!l || $l) return "";
    $l = !0;
    var u = Error.prepareStackTrace;
    Error.prepareStackTrace = void 0;
    try {
      var c = {
        DetermineComponentFrameRoot: function() {
          try {
            if (n) {
              var $ = function() {
                throw Error();
              };
              if (Object.defineProperty($.prototype, "props", {
                set: function() {
                  throw Error();
                }
              }), typeof Reflect == "object" && Reflect.construct) {
                try {
                  Reflect.construct($, []);
                } catch (Y) {
                  var q = Y;
                }
                Reflect.construct(l, [], $);
              } else {
                try {
                  $.call();
                } catch (Y) {
                  q = Y;
                }
                l.call($.prototype);
              }
            } else {
              try {
                throw Error();
              } catch (Y) {
                q = Y;
              }
              ($ = l()) && typeof $.catch == "function" && $.catch(function() {
              });
            }
          } catch (Y) {
            if (Y && q && typeof Y.stack == "string")
              return [Y.stack, q.stack];
          }
          return [null, null];
        }
      };
      c.DetermineComponentFrameRoot.displayName = "DetermineComponentFrameRoot";
      var s = Object.getOwnPropertyDescriptor(
        c.DetermineComponentFrameRoot,
        "name"
      );
      s && s.configurable && Object.defineProperty(
        c.DetermineComponentFrameRoot,
        "name",
        { value: "DetermineComponentFrameRoot" }
      );
      var r = c.DetermineComponentFrameRoot(), y = r[0], p = r[1];
      if (y && p) {
        var S = y.split(`
`), H = p.split(`
`);
        for (s = c = 0; c < S.length && !S[c].includes("DetermineComponentFrameRoot"); )
          c++;
        for (; s < H.length && !H[s].includes(
          "DetermineComponentFrameRoot"
        ); )
          s++;
        if (c === S.length || s === H.length)
          for (c = S.length - 1, s = H.length - 1; 1 <= c && 0 <= s && S[c] !== H[s]; )
            s--;
        for (; 1 <= c && 0 <= s; c--, s--)
          if (S[c] !== H[s]) {
            if (c !== 1 || s !== 1)
              do
                if (c--, s--, 0 > s || S[c] !== H[s]) {
                  var K = `
` + S[c].replace(" at new ", " at ");
                  return l.displayName && K.includes("<anonymous>") && (K = K.replace("<anonymous>", l.displayName)), K;
                }
              while (1 <= c && 0 <= s);
            break;
          }
      }
    } finally {
      $l = !1, Error.prepareStackTrace = u;
    }
    return (u = l ? l.displayName || l.name : "") ? Ji(u) : "";
  }
  function ki(l) {
    switch (l.tag) {
      case 26:
      case 27:
      case 5:
        return Ji(l.type);
      case 16:
        return Ji("Lazy");
      case 13:
        return Ji("Suspense");
      case 19:
        return Ji("SuspenseList");
      case 0:
      case 15:
        return li(l.type, !1);
      case 11:
        return li(l.type.render, !1);
      case 1:
        return li(l.type, !0);
      case 31:
        return Ji("Activity");
      default:
        return "";
    }
  }
  function wh(l) {
    try {
      var n = "";
      do
        n += ki(l), l = l.return;
      while (l);
      return n;
    } catch (u) {
      return `
Error generating stack: ` + u.message + `
` + u.stack;
    }
  }
  function Gl(l) {
    switch (typeof l) {
      case "bigint":
      case "boolean":
      case "number":
      case "string":
      case "undefined":
        return l;
      case "object":
        return l;
      default:
        return "";
    }
  }
  function Df(l) {
    var n = l.type;
    return (l = l.nodeName) && l.toLowerCase() === "input" && (n === "checkbox" || n === "radio");
  }
  function qh(l) {
    var n = Df(l) ? "checked" : "value", u = Object.getOwnPropertyDescriptor(
      l.constructor.prototype,
      n
    ), c = "" + l[n];
    if (!l.hasOwnProperty(n) && typeof u < "u" && typeof u.get == "function" && typeof u.set == "function") {
      var s = u.get, r = u.set;
      return Object.defineProperty(l, n, {
        configurable: !0,
        get: function() {
          return s.call(this);
        },
        set: function(y) {
          c = "" + y, r.call(this, y);
        }
      }), Object.defineProperty(l, n, {
        enumerable: u.enumerable
      }), {
        getValue: function() {
          return c;
        },
        setValue: function(y) {
          c = "" + y;
        },
        stopTracking: function() {
          l._valueTracker = null, delete l[n];
        }
      };
    }
  }
  function ai(l) {
    l._valueTracker || (l._valueTracker = qh(l));
  }
  function $i(l) {
    if (!l) return !1;
    var n = l._valueTracker;
    if (!n) return !0;
    var u = n.getValue(), c = "";
    return l && (c = Df(l) ? l.checked ? "true" : "false" : l.value), l = c, l !== u ? (n.setValue(l), !0) : !1;
  }
  function no(l) {
    if (l = l || (typeof document < "u" ? document : void 0), typeof l > "u") return null;
    try {
      return l.activeElement || l.body;
    } catch {
      return l.body;
    }
  }
  var Eg = /[\n"\\]/g;
  function Ya(l) {
    return l.replace(
      Eg,
      function(n) {
        return "\\" + n.charCodeAt(0).toString(16) + " ";
      }
    );
  }
  function hr(l, n, u, c, s, r, y, p) {
    l.name = "", y != null && typeof y != "function" && typeof y != "symbol" && typeof y != "boolean" ? l.type = y : l.removeAttribute("type"), n != null ? y === "number" ? (n === 0 && l.value === "" || l.value != n) && (l.value = "" + Gl(n)) : l.value !== "" + Gl(n) && (l.value = "" + Gl(n)) : y !== "submit" && y !== "reset" || l.removeAttribute("value"), n != null ? zf(l, y, Gl(n)) : u != null ? zf(l, y, Gl(u)) : c != null && l.removeAttribute("value"), s == null && r != null && (l.defaultChecked = !!r), s != null && (l.checked = s && typeof s != "function" && typeof s != "symbol"), p != null && typeof p != "function" && typeof p != "symbol" && typeof p != "boolean" ? l.name = "" + Gl(p) : l.removeAttribute("name");
  }
  function yr(l, n, u, c, s, r, y, p) {
    if (r != null && typeof r != "function" && typeof r != "symbol" && typeof r != "boolean" && (l.type = r), n != null || u != null) {
      if (!(r !== "submit" && r !== "reset" || n != null))
        return;
      u = u != null ? "" + Gl(u) : "", n = n != null ? "" + Gl(n) : u, p || n === l.value || (l.value = n), l.defaultValue = n;
    }
    c = c ?? s, c = typeof c != "function" && typeof c != "symbol" && !!c, l.checked = p ? l.checked : !!c, l.defaultChecked = !!c, y != null && typeof y != "function" && typeof y != "symbol" && typeof y != "boolean" && (l.name = y);
  }
  function zf(l, n, u) {
    n === "number" && no(l.ownerDocument) === l || l.defaultValue === "" + u || (l.defaultValue = "" + u);
  }
  function Wi(l, n, u, c) {
    if (l = l.options, n) {
      n = {};
      for (var s = 0; s < u.length; s++)
        n["$" + u[s]] = !0;
      for (u = 0; u < l.length; u++)
        s = n.hasOwnProperty("$" + l[u].value), l[u].selected !== s && (l[u].selected = s), s && c && (l[u].defaultSelected = !0);
    } else {
      for (u = "" + Gl(u), n = null, s = 0; s < l.length; s++) {
        if (l[s].value === u) {
          l[s].selected = !0, c && (l[s].defaultSelected = !0);
          return;
        }
        n !== null || l[s].disabled || (n = l[s]);
      }
      n !== null && (n.selected = !0);
    }
  }
  function jh(l, n, u) {
    if (n != null && (n = "" + Gl(n), n !== l.value && (l.value = n), u == null)) {
      l.defaultValue !== n && (l.defaultValue = n);
      return;
    }
    l.defaultValue = u != null ? "" + Gl(u) : "";
  }
  function Bh(l, n, u, c) {
    if (n == null) {
      if (c != null) {
        if (u != null) throw Error(_(92));
        if (pt(c)) {
          if (1 < c.length) throw Error(_(93));
          c = c[0];
        }
        u = c;
      }
      u == null && (u = ""), n = u;
    }
    u = Gl(n), l.defaultValue = u, c = l.textContent, c === u && c !== "" && c !== null && (l.value = c);
  }
  function uo(l, n) {
    if (n) {
      var u = l.firstChild;
      if (u && u === l.lastChild && u.nodeType === 3) {
        u.nodeValue = n;
        return;
      }
    }
    l.textContent = n;
  }
  var Up = new Set(
    "animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(
      " "
    )
  );
  function mr(l, n, u) {
    var c = n.indexOf("--") === 0;
    u == null || typeof u == "boolean" || u === "" ? c ? l.setProperty(n, "") : n === "float" ? l.cssFloat = "" : l[n] = "" : c ? l.setProperty(n, u) : typeof u != "number" || u === 0 || Up.has(n) ? n === "float" ? l.cssFloat = u : l[n] = ("" + u).trim() : l[n] = u + "px";
  }
  function Mf(l, n, u) {
    if (n != null && typeof n != "object")
      throw Error(_(62));
    if (l = l.style, u != null) {
      for (var c in u)
        !u.hasOwnProperty(c) || n != null && n.hasOwnProperty(c) || (c.indexOf("--") === 0 ? l.setProperty(c, "") : c === "float" ? l.cssFloat = "" : l[c] = "");
      for (var s in n)
        c = n[s], n.hasOwnProperty(s) && u[s] !== c && mr(l, s, c);
    } else
      for (var r in n)
        n.hasOwnProperty(r) && mr(l, r, n[r]);
  }
  function Fi(l) {
    if (l.indexOf("-") === -1) return !1;
    switch (l) {
      case "annotation-xml":
      case "color-profile":
      case "font-face":
      case "font-face-src":
      case "font-face-uri":
      case "font-face-format":
      case "font-face-name":
      case "missing-glyph":
        return !1;
      default:
        return !0;
    }
  }
  var Ag = /* @__PURE__ */ new Map([
    ["acceptCharset", "accept-charset"],
    ["htmlFor", "for"],
    ["httpEquiv", "http-equiv"],
    ["crossOrigin", "crossorigin"],
    ["accentHeight", "accent-height"],
    ["alignmentBaseline", "alignment-baseline"],
    ["arabicForm", "arabic-form"],
    ["baselineShift", "baseline-shift"],
    ["capHeight", "cap-height"],
    ["clipPath", "clip-path"],
    ["clipRule", "clip-rule"],
    ["colorInterpolation", "color-interpolation"],
    ["colorInterpolationFilters", "color-interpolation-filters"],
    ["colorProfile", "color-profile"],
    ["colorRendering", "color-rendering"],
    ["dominantBaseline", "dominant-baseline"],
    ["enableBackground", "enable-background"],
    ["fillOpacity", "fill-opacity"],
    ["fillRule", "fill-rule"],
    ["floodColor", "flood-color"],
    ["floodOpacity", "flood-opacity"],
    ["fontFamily", "font-family"],
    ["fontSize", "font-size"],
    ["fontSizeAdjust", "font-size-adjust"],
    ["fontStretch", "font-stretch"],
    ["fontStyle", "font-style"],
    ["fontVariant", "font-variant"],
    ["fontWeight", "font-weight"],
    ["glyphName", "glyph-name"],
    ["glyphOrientationHorizontal", "glyph-orientation-horizontal"],
    ["glyphOrientationVertical", "glyph-orientation-vertical"],
    ["horizAdvX", "horiz-adv-x"],
    ["horizOriginX", "horiz-origin-x"],
    ["imageRendering", "image-rendering"],
    ["letterSpacing", "letter-spacing"],
    ["lightingColor", "lighting-color"],
    ["markerEnd", "marker-end"],
    ["markerMid", "marker-mid"],
    ["markerStart", "marker-start"],
    ["overlinePosition", "overline-position"],
    ["overlineThickness", "overline-thickness"],
    ["paintOrder", "paint-order"],
    ["panose-1", "panose-1"],
    ["pointerEvents", "pointer-events"],
    ["renderingIntent", "rendering-intent"],
    ["shapeRendering", "shape-rendering"],
    ["stopColor", "stop-color"],
    ["stopOpacity", "stop-opacity"],
    ["strikethroughPosition", "strikethrough-position"],
    ["strikethroughThickness", "strikethrough-thickness"],
    ["strokeDasharray", "stroke-dasharray"],
    ["strokeDashoffset", "stroke-dashoffset"],
    ["strokeLinecap", "stroke-linecap"],
    ["strokeLinejoin", "stroke-linejoin"],
    ["strokeMiterlimit", "stroke-miterlimit"],
    ["strokeOpacity", "stroke-opacity"],
    ["strokeWidth", "stroke-width"],
    ["textAnchor", "text-anchor"],
    ["textDecoration", "text-decoration"],
    ["textRendering", "text-rendering"],
    ["transformOrigin", "transform-origin"],
    ["underlinePosition", "underline-position"],
    ["underlineThickness", "underline-thickness"],
    ["unicodeBidi", "unicode-bidi"],
    ["unicodeRange", "unicode-range"],
    ["unitsPerEm", "units-per-em"],
    ["vAlphabetic", "v-alphabetic"],
    ["vHanging", "v-hanging"],
    ["vIdeographic", "v-ideographic"],
    ["vMathematical", "v-mathematical"],
    ["vectorEffect", "vector-effect"],
    ["vertAdvY", "vert-adv-y"],
    ["vertOriginX", "vert-origin-x"],
    ["vertOriginY", "vert-origin-y"],
    ["wordSpacing", "word-spacing"],
    ["writingMode", "writing-mode"],
    ["xmlnsXlink", "xmlns:xlink"],
    ["xHeight", "x-height"]
  ]), Cp = /^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;
  function _f(l) {
    return Cp.test("" + l) ? "javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')" : l;
  }
  var Ii = null;
  function pr(l) {
    return l = l.target || l.srcElement || window, l.correspondingUseElement && (l = l.correspondingUseElement), l.nodeType === 3 ? l.parentNode : l;
  }
  var io = null, co = null;
  function xp(l) {
    var n = Ki(l);
    if (n && (l = n.stateNode)) {
      var u = l[kl] || null;
      e: switch (l = n.stateNode, n.type) {
        case "input":
          if (hr(
            l,
            u.value,
            u.defaultValue,
            u.defaultValue,
            u.checked,
            u.defaultChecked,
            u.type,
            u.name
          ), n = u.name, u.type === "radio" && n != null) {
            for (u = l; u.parentNode; ) u = u.parentNode;
            for (u = u.querySelectorAll(
              'input[name="' + Ya(
                "" + n
              ) + '"][type="radio"]'
            ), n = 0; n < u.length; n++) {
              var c = u[n];
              if (c !== l && c.form === l.form) {
                var s = c[kl] || null;
                if (!s) throw Error(_(90));
                hr(
                  c,
                  s.value,
                  s.defaultValue,
                  s.defaultValue,
                  s.checked,
                  s.defaultChecked,
                  s.type,
                  s.name
                );
              }
            }
            for (n = 0; n < u.length; n++)
              c = u[n], c.form === l.form && $i(c);
          }
          break e;
        case "textarea":
          jh(l, u.value, u.defaultValue);
          break e;
        case "select":
          n = u.value, n != null && Wi(l, !!u.multiple, n, !1);
      }
    }
  }
  var Yh = !1;
  function oo(l, n, u) {
    if (Yh) return l(n, u);
    Yh = !0;
    try {
      var c = l(n);
      return c;
    } finally {
      if (Yh = !1, (io !== null || co !== null) && (Uc(), io && (n = io, l = co, co = io = null, xp(n), l)))
        for (n = 0; n < l.length; n++) xp(l[n]);
    }
  }
  function Pi(l, n) {
    var u = l.stateNode;
    if (u === null) return null;
    var c = u[kl] || null;
    if (c === null) return null;
    u = c[n];
    e: switch (n) {
      case "onClick":
      case "onClickCapture":
      case "onDoubleClick":
      case "onDoubleClickCapture":
      case "onMouseDown":
      case "onMouseDownCapture":
      case "onMouseMove":
      case "onMouseMoveCapture":
      case "onMouseUp":
      case "onMouseUpCapture":
      case "onMouseEnter":
        (c = !c.disabled) || (l = l.type, c = !(l === "button" || l === "input" || l === "select" || l === "textarea")), l = !c;
        break e;
      default:
        l = !1;
    }
    if (l) return null;
    if (u && typeof u != "function")
      throw Error(
        _(231, n, typeof u)
      );
    return u;
  }
  var Un = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), vr = !1;
  if (Un)
    try {
      var Tu = {};
      Object.defineProperty(Tu, "passive", {
        get: function() {
          vr = !0;
        }
      }), window.addEventListener("test", Tu, Tu), window.removeEventListener("test", Tu, Tu);
    } catch {
      vr = !1;
    }
  var Eu = null, fo = null, ec = null;
  function Gh() {
    if (ec) return ec;
    var l, n = fo, u = n.length, c, s = "value" in Eu ? Eu.value : Eu.textContent, r = s.length;
    for (l = 0; l < u && n[l] === s[l]; l++) ;
    var y = u - l;
    for (c = 1; c <= y && n[u - c] === s[r - c]; c++) ;
    return ec = s.slice(l, 1 < c ? 1 - c : void 0);
  }
  function _l(l) {
    var n = l.keyCode;
    return "charCode" in l ? (l = l.charCode, l === 0 && n === 13 && (l = 13)) : l = n, l === 10 && (l = 13), 32 <= l || l === 13 ? l : 0;
  }
  function gr() {
    return !0;
  }
  function br() {
    return !1;
  }
  function Wl(l) {
    function n(u, c, s, r, y) {
      this._reactName = u, this._targetInst = s, this.type = c, this.nativeEvent = r, this.target = y, this.currentTarget = null;
      for (var p in l)
        l.hasOwnProperty(p) && (u = l[p], this[p] = u ? u(r) : r[p]);
      return this.isDefaultPrevented = (r.defaultPrevented != null ? r.defaultPrevented : r.returnValue === !1) ? gr : br, this.isPropagationStopped = br, this;
    }
    return te(n.prototype, {
      preventDefault: function() {
        this.defaultPrevented = !0;
        var u = this.nativeEvent;
        u && (u.preventDefault ? u.preventDefault() : typeof u.returnValue != "unknown" && (u.returnValue = !1), this.isDefaultPrevented = gr);
      },
      stopPropagation: function() {
        var u = this.nativeEvent;
        u && (u.stopPropagation ? u.stopPropagation() : typeof u.cancelBubble != "unknown" && (u.cancelBubble = !0), this.isPropagationStopped = gr);
      },
      persist: function() {
      },
      isPersistent: gr
    }), n;
  }
  var ni = {
    eventPhase: 0,
    bubbles: 0,
    cancelable: 0,
    timeStamp: function(l) {
      return l.timeStamp || Date.now();
    },
    defaultPrevented: 0,
    isTrusted: 0
  }, Sr = Wl(ni), Uf = te({}, ni, { view: 0, detail: 0 }), Hp = Wl(Uf), Lh, Tr, Cf, tc = te({}, Uf, {
    screenX: 0,
    screenY: 0,
    clientX: 0,
    clientY: 0,
    pageX: 0,
    pageY: 0,
    ctrlKey: 0,
    shiftKey: 0,
    altKey: 0,
    metaKey: 0,
    getModifierState: Au,
    button: 0,
    buttons: 0,
    relatedTarget: function(l) {
      return l.relatedTarget === void 0 ? l.fromElement === l.srcElement ? l.toElement : l.fromElement : l.relatedTarget;
    },
    movementX: function(l) {
      return "movementX" in l ? l.movementX : (l !== Cf && (Cf && l.type === "mousemove" ? (Lh = l.screenX - Cf.screenX, Tr = l.screenY - Cf.screenY) : Tr = Lh = 0, Cf = l), Lh);
    },
    movementY: function(l) {
      return "movementY" in l ? l.movementY : Tr;
    }
  }), Vh = Wl(tc), Np = te({}, tc, { dataTransfer: 0 }), wp = Wl(Np), Rg = te({}, Uf, { relatedTarget: 0 }), Xh = Wl(Rg), Og = te({}, ni, {
    animationName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), Dg = Wl(Og), zg = te({}, ni, {
    clipboardData: function(l) {
      return "clipboardData" in l ? l.clipboardData : window.clipboardData;
    }
  }), xf = Wl(zg), qp = te({}, ni, { data: 0 }), Qh = Wl(qp), jp = {
    Esc: "Escape",
    Spacebar: " ",
    Left: "ArrowLeft",
    Up: "ArrowUp",
    Right: "ArrowRight",
    Down: "ArrowDown",
    Del: "Delete",
    Win: "OS",
    Menu: "ContextMenu",
    Apps: "ContextMenu",
    Scroll: "ScrollLock",
    MozPrintableKey: "Unidentified"
  }, Bp = {
    8: "Backspace",
    9: "Tab",
    12: "Clear",
    13: "Enter",
    16: "Shift",
    17: "Control",
    18: "Alt",
    19: "Pause",
    20: "CapsLock",
    27: "Escape",
    32: " ",
    33: "PageUp",
    34: "PageDown",
    35: "End",
    36: "Home",
    37: "ArrowLeft",
    38: "ArrowUp",
    39: "ArrowRight",
    40: "ArrowDown",
    45: "Insert",
    46: "Delete",
    112: "F1",
    113: "F2",
    114: "F3",
    115: "F4",
    116: "F5",
    117: "F6",
    118: "F7",
    119: "F8",
    120: "F9",
    121: "F10",
    122: "F11",
    123: "F12",
    144: "NumLock",
    145: "ScrollLock",
    224: "Meta"
  }, Zh = {
    Alt: "altKey",
    Control: "ctrlKey",
    Meta: "metaKey",
    Shift: "shiftKey"
  };
  function Yp(l) {
    var n = this.nativeEvent;
    return n.getModifierState ? n.getModifierState(l) : (l = Zh[l]) ? !!n[l] : !1;
  }
  function Au() {
    return Yp;
  }
  var lc = te({}, Uf, {
    key: function(l) {
      if (l.key) {
        var n = jp[l.key] || l.key;
        if (n !== "Unidentified") return n;
      }
      return l.type === "keypress" ? (l = _l(l), l === 13 ? "Enter" : String.fromCharCode(l)) : l.type === "keydown" || l.type === "keyup" ? Bp[l.keyCode] || "Unidentified" : "";
    },
    code: 0,
    location: 0,
    ctrlKey: 0,
    shiftKey: 0,
    altKey: 0,
    metaKey: 0,
    repeat: 0,
    locale: 0,
    getModifierState: Au,
    charCode: function(l) {
      return l.type === "keypress" ? _l(l) : 0;
    },
    keyCode: function(l) {
      return l.type === "keydown" || l.type === "keyup" ? l.keyCode : 0;
    },
    which: function(l) {
      return l.type === "keypress" ? _l(l) : l.type === "keydown" || l.type === "keyup" ? l.keyCode : 0;
    }
  }), on = Wl(lc), Ra = te({}, tc, {
    pointerId: 0,
    width: 0,
    height: 0,
    pressure: 0,
    tangentialPressure: 0,
    tiltX: 0,
    tiltY: 0,
    twist: 0,
    pointerType: 0,
    isPrimary: 0
  }), Hf = Wl(Ra), Er = te({}, Uf, {
    touches: 0,
    targetTouches: 0,
    changedTouches: 0,
    altKey: 0,
    metaKey: 0,
    ctrlKey: 0,
    shiftKey: 0,
    getModifierState: Au
  }), Kh = Wl(Er), ia = te({}, ni, {
    propertyName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), Gp = Wl(ia), Ar = te({}, tc, {
    deltaX: function(l) {
      return "deltaX" in l ? l.deltaX : "wheelDeltaX" in l ? -l.wheelDeltaX : 0;
    },
    deltaY: function(l) {
      return "deltaY" in l ? l.deltaY : "wheelDeltaY" in l ? -l.wheelDeltaY : "wheelDelta" in l ? -l.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), ac = Wl(Ar), Jh = te({}, ni, {
    newState: 0,
    oldState: 0
  }), Lp = Wl(Jh), Vp = [9, 13, 27, 32], Nf = Un && "CompositionEvent" in window, wf = null;
  Un && "documentMode" in document && (wf = document.documentMode);
  var kh = Un && "TextEvent" in window && !wf, Cn = Un && (!Nf || wf && 8 < wf && 11 >= wf), $h = " ", Rr = !1;
  function qf(l, n) {
    switch (l) {
      case "keyup":
        return Vp.indexOf(n.keyCode) !== -1;
      case "keydown":
        return n.keyCode !== 229;
      case "keypress":
      case "mousedown":
      case "focusout":
        return !0;
      default:
        return !1;
    }
  }
  function ui(l) {
    return l = l.detail, typeof l == "object" && "data" in l ? l.data : null;
  }
  var ii = !1;
  function Wh(l, n) {
    switch (l) {
      case "compositionend":
        return ui(n);
      case "keypress":
        return n.which !== 32 ? null : (Rr = !0, $h);
      case "textInput":
        return l = n.data, l === $h && Rr ? null : l;
      default:
        return null;
    }
  }
  function nc(l, n) {
    if (ii)
      return l === "compositionend" || !Nf && qf(l, n) ? (l = Gh(), ec = fo = Eu = null, ii = !1, l) : null;
    switch (l) {
      case "paste":
        return null;
      case "keypress":
        if (!(n.ctrlKey || n.altKey || n.metaKey) || n.ctrlKey && n.altKey) {
          if (n.char && 1 < n.char.length)
            return n.char;
          if (n.which) return String.fromCharCode(n.which);
        }
        return null;
      case "compositionend":
        return Cn && n.locale !== "ko" ? null : n.data;
      default:
        return null;
    }
  }
  var Xp = {
    color: !0,
    date: !0,
    datetime: !0,
    "datetime-local": !0,
    email: !0,
    month: !0,
    number: !0,
    password: !0,
    range: !0,
    search: !0,
    tel: !0,
    text: !0,
    time: !0,
    url: !0,
    week: !0
  };
  function Or(l) {
    var n = l && l.nodeName && l.nodeName.toLowerCase();
    return n === "input" ? !!Xp[l.type] : n === "textarea";
  }
  function Dr(l, n, u, c) {
    io ? co ? co.push(c) : co = [c] : io = c, n = Jo(n, "onChange"), 0 < n.length && (u = new Sr(
      "onChange",
      "change",
      null,
      u,
      c
    ), l.push({ event: u, listeners: n }));
  }
  var fn = null, sn = null;
  function Fh(l) {
    Nc(l, 0);
  }
  function xn(l) {
    var n = Af(l);
    if ($i(n)) return l;
  }
  function Ih(l, n) {
    if (l === "change") return n;
  }
  var Ph = !1;
  if (Un) {
    var uc;
    if (Un) {
      var ic = "oninput" in document;
      if (!ic) {
        var ey = document.createElement("div");
        ey.setAttribute("oninput", "return;"), ic = typeof ey.oninput == "function";
      }
      uc = ic;
    } else uc = !1;
    Ph = uc && (!document.documentMode || 9 < document.documentMode);
  }
  function so() {
    fn && (fn.detachEvent("onpropertychange", ty), sn = fn = null);
  }
  function ty(l) {
    if (l.propertyName === "value" && xn(sn)) {
      var n = [];
      Dr(
        n,
        sn,
        l,
        pr(l)
      ), oo(Fh, n);
    }
  }
  function zr(l, n, u) {
    l === "focusin" ? (so(), fn = n, sn = u, fn.attachEvent("onpropertychange", ty)) : l === "focusout" && so();
  }
  function ci(l) {
    if (l === "selectionchange" || l === "keyup" || l === "keydown")
      return xn(sn);
  }
  function Ru(l, n) {
    if (l === "click") return xn(n);
  }
  function ly(l, n) {
    if (l === "input" || l === "change")
      return xn(n);
  }
  function ay(l, n) {
    return l === n && (l !== 0 || 1 / l === 1 / n) || l !== l && n !== n;
  }
  var Ul = typeof Object.is == "function" ? Object.is : ay;
  function oi(l, n) {
    if (Ul(l, n)) return !0;
    if (typeof l != "object" || l === null || typeof n != "object" || n === null)
      return !1;
    var u = Object.keys(l), c = Object.keys(n);
    if (u.length !== c.length) return !1;
    for (c = 0; c < u.length; c++) {
      var s = u[c];
      if (!Zi.call(n, s) || !Ul(l[s], n[s]))
        return !1;
    }
    return !0;
  }
  function fi(l) {
    for (; l && l.firstChild; ) l = l.firstChild;
    return l;
  }
  function Ct(l, n) {
    var u = fi(l);
    l = 0;
    for (var c; u; ) {
      if (u.nodeType === 3) {
        if (c = l + u.textContent.length, l <= n && c >= n)
          return { node: u, offset: n - l };
        l = c;
      }
      e: {
        for (; u; ) {
          if (u.nextSibling) {
            u = u.nextSibling;
            break e;
          }
          u = u.parentNode;
        }
        u = void 0;
      }
      u = fi(u);
    }
  }
  function jf(l, n) {
    return l && n ? l === n ? !0 : l && l.nodeType === 3 ? !1 : n && n.nodeType === 3 ? jf(l, n.parentNode) : "contains" in l ? l.contains(n) : l.compareDocumentPosition ? !!(l.compareDocumentPosition(n) & 16) : !1 : !1;
  }
  function ny(l) {
    l = l != null && l.ownerDocument != null && l.ownerDocument.defaultView != null ? l.ownerDocument.defaultView : window;
    for (var n = no(l.document); n instanceof l.HTMLIFrameElement; ) {
      try {
        var u = typeof n.contentWindow.location.href == "string";
      } catch {
        u = !1;
      }
      if (u) l = n.contentWindow;
      else break;
      n = no(l.document);
    }
    return n;
  }
  function Bf(l) {
    var n = l && l.nodeName && l.nodeName.toLowerCase();
    return n && (n === "input" && (l.type === "text" || l.type === "search" || l.type === "tel" || l.type === "url" || l.type === "password") || n === "textarea" || l.contentEditable === "true");
  }
  var cc = Un && "documentMode" in document && 11 >= document.documentMode, Hn = null, rn = null, si = null, oc = !1;
  function Mr(l, n, u) {
    var c = u.window === u ? u.document : u.nodeType === 9 ? u : u.ownerDocument;
    oc || Hn == null || Hn !== no(c) || (c = Hn, "selectionStart" in c && Bf(c) ? c = { start: c.selectionStart, end: c.selectionEnd } : (c = (c.ownerDocument && c.ownerDocument.defaultView || window).getSelection(), c = {
      anchorNode: c.anchorNode,
      anchorOffset: c.anchorOffset,
      focusNode: c.focusNode,
      focusOffset: c.focusOffset
    }), si && oi(si, c) || (si = c, c = Jo(rn, "onSelect"), 0 < c.length && (n = new Sr(
      "onSelect",
      "select",
      null,
      n,
      u
    ), l.push({ event: n, listeners: c }), n.target = Hn)));
  }
  function Ou(l, n) {
    var u = {};
    return u[l.toLowerCase()] = n.toLowerCase(), u["Webkit" + l] = "webkit" + n, u["Moz" + l] = "moz" + n, u;
  }
  var fc = {
    animationend: Ou("Animation", "AnimationEnd"),
    animationiteration: Ou("Animation", "AnimationIteration"),
    animationstart: Ou("Animation", "AnimationStart"),
    transitionrun: Ou("Transition", "TransitionRun"),
    transitionstart: Ou("Transition", "TransitionStart"),
    transitioncancel: Ou("Transition", "TransitionCancel"),
    transitionend: Ou("Transition", "TransitionEnd")
  }, Ga = {}, dn = {};
  Un && (dn = document.createElement("div").style, "AnimationEvent" in window || (delete fc.animationend.animation, delete fc.animationiteration.animation, delete fc.animationstart.animation), "TransitionEvent" in window || delete fc.transitionend.transition);
  function Nn(l) {
    if (Ga[l]) return Ga[l];
    if (!fc[l]) return l;
    var n = fc[l], u;
    for (u in n)
      if (n.hasOwnProperty(u) && u in dn)
        return Ga[l] = n[u];
    return l;
  }
  var Qp = Nn("animationend"), uy = Nn("animationiteration"), Zp = Nn("animationstart"), iy = Nn("transitionrun"), _r = Nn("transitionstart"), Kp = Nn("transitioncancel"), cy = Nn("transitionend"), oy = /* @__PURE__ */ new Map(), ro = "abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(
    " "
  );
  ro.push("scrollEnd");
  function La(l, n) {
    oy.set(l, n), ei(n, [l]);
  }
  var fy = /* @__PURE__ */ new WeakMap();
  function Oa(l, n) {
    if (typeof l == "object" && l !== null) {
      var u = fy.get(l);
      return u !== void 0 ? u : (n = {
        value: l,
        source: n,
        stack: wh(n)
      }, fy.set(l, n), n);
    }
    return {
      value: l,
      source: n,
      stack: wh(n)
    };
  }
  var ca = [], ri = 0, wn = 0;
  function hn() {
    for (var l = ri, n = wn = ri = 0; n < l; ) {
      var u = ca[n];
      ca[n++] = null;
      var c = ca[n];
      ca[n++] = null;
      var s = ca[n];
      ca[n++] = null;
      var r = ca[n];
      if (ca[n++] = null, c !== null && s !== null) {
        var y = c.pending;
        y === null ? s.next = s : (s.next = y.next, y.next = s), c.pending = s;
      }
      r !== 0 && yo(u, s, r);
    }
  }
  function di(l, n, u, c) {
    ca[ri++] = l, ca[ri++] = n, ca[ri++] = u, ca[ri++] = c, wn |= c, l.lanes |= c, l = l.alternate, l !== null && (l.lanes |= c);
  }
  function ho(l, n, u, c) {
    return di(l, n, u, c), Yf(l);
  }
  function qn(l, n) {
    return di(l, null, null, n), Yf(l);
  }
  function yo(l, n, u) {
    l.lanes |= u;
    var c = l.alternate;
    c !== null && (c.lanes |= u);
    for (var s = !1, r = l.return; r !== null; )
      r.childLanes |= u, c = r.alternate, c !== null && (c.childLanes |= u), r.tag === 22 && (l = r.stateNode, l === null || l._visibility & 1 || (s = !0)), l = r, r = r.return;
    return l.tag === 3 ? (r = l.stateNode, s && n !== null && (s = 31 - Dl(u), l = r.hiddenUpdates, c = l[s], c === null ? l[s] = [n] : c.push(n), n.lane = u | 536870912), r) : null;
  }
  function Yf(l) {
    if (50 < Lo)
      throw Lo = 0, fm = null, Error(_(185));
    for (var n = l.return; n !== null; )
      l = n, n = l.return;
    return l.tag === 3 ? l.stateNode : null;
  }
  var mo = {};
  function Jp(l, n, u, c) {
    this.tag = l, this.key = u, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.refCleanup = this.ref = null, this.pendingProps = n, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = c, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function oa(l, n, u, c) {
    return new Jp(l, n, u, c);
  }
  function Gf(l) {
    return l = l.prototype, !(!l || !l.isReactComponent);
  }
  function yn(l, n) {
    var u = l.alternate;
    return u === null ? (u = oa(
      l.tag,
      n,
      l.key,
      l.mode
    ), u.elementType = l.elementType, u.type = l.type, u.stateNode = l.stateNode, u.alternate = l, l.alternate = u) : (u.pendingProps = n, u.type = l.type, u.flags = 0, u.subtreeFlags = 0, u.deletions = null), u.flags = l.flags & 65011712, u.childLanes = l.childLanes, u.lanes = l.lanes, u.child = l.child, u.memoizedProps = l.memoizedProps, u.memoizedState = l.memoizedState, u.updateQueue = l.updateQueue, n = l.dependencies, u.dependencies = n === null ? null : { lanes: n.lanes, firstContext: n.firstContext }, u.sibling = l.sibling, u.index = l.index, u.ref = l.ref, u.refCleanup = l.refCleanup, u;
  }
  function $e(l, n) {
    l.flags &= 65011714;
    var u = l.alternate;
    return u === null ? (l.childLanes = 0, l.lanes = n, l.child = null, l.subtreeFlags = 0, l.memoizedProps = null, l.memoizedState = null, l.updateQueue = null, l.dependencies = null, l.stateNode = null) : (l.childLanes = u.childLanes, l.lanes = u.lanes, l.child = u.child, l.subtreeFlags = 0, l.deletions = null, l.memoizedProps = u.memoizedProps, l.memoizedState = u.memoizedState, l.updateQueue = u.updateQueue, l.type = u.type, n = u.dependencies, l.dependencies = n === null ? null : {
      lanes: n.lanes,
      firstContext: n.firstContext
    }), l;
  }
  function ee(l, n, u, c, s, r) {
    var y = 0;
    if (c = l, typeof l == "function") Gf(l) && (y = 1);
    else if (typeof l == "string")
      y = zv(
        l,
        u,
        ce.current
      ) ? 26 : l === "html" || l === "head" || l === "body" ? 27 : 5;
    else
      e: switch (l) {
        case re:
          return l = oa(31, u, n, s), l.elementType = re, l.lanes = r, l;
        case je:
          return Va(u.children, s, r, n);
        case At:
          y = 8, s |= 24;
          break;
        case ke:
          return l = oa(12, u, n, s | 2), l.elementType = ke, l.lanes = r, l;
        case se:
          return l = oa(13, u, n, s), l.elementType = se, l.lanes = r, l;
        case Ue:
          return l = oa(19, u, n, s), l.elementType = Ue, l.lanes = r, l;
        default:
          if (typeof l == "object" && l !== null)
            switch (l.$$typeof) {
              case nt:
              case it:
                y = 10;
                break e;
              case el:
                y = 9;
                break e;
              case Nt:
                y = 11;
                break e;
              case De:
                y = 14;
                break e;
              case bt:
                y = 16, c = null;
                break e;
            }
          y = 29, u = Error(
            _(130, l === null ? "null" : typeof l, "")
          ), c = null;
      }
    return n = oa(y, u, n, s), n.elementType = l, n.type = c, n.lanes = r, n;
  }
  function Va(l, n, u, c) {
    return l = oa(7, l, c, n), l.lanes = u, l;
  }
  function po(l, n, u) {
    return l = oa(6, l, null, n), l.lanes = u, l;
  }
  function Lt(l, n, u) {
    return n = oa(
      4,
      l.children !== null ? l.children : [],
      l.key,
      n
    ), n.lanes = u, n.stateNode = {
      containerInfo: l.containerInfo,
      pendingChildren: null,
      implementation: l.implementation
    }, n;
  }
  var hi = [], yi = 0, Lf = null, vo = 0, Xa = [], fa = 0, Du = null, mn = 1, Qt = "";
  function ot(l, n) {
    hi[yi++] = vo, hi[yi++] = Lf, Lf = l, vo = n;
  }
  function Ur(l, n, u) {
    Xa[fa++] = mn, Xa[fa++] = Qt, Xa[fa++] = Du, Du = l;
    var c = mn;
    l = Qt;
    var s = 32 - Dl(c) - 1;
    c &= ~(1 << s), u += 1;
    var r = 32 - Dl(n) + s;
    if (30 < r) {
      var y = s - s % 5;
      r = (c & (1 << y) - 1).toString(32), c >>= y, s -= y, mn = 1 << 32 - Dl(n) + s | u << s | c, Qt = r + l;
    } else
      mn = 1 << r | u << s | c, Qt = l;
  }
  function sc(l) {
    l.return !== null && (ot(l, 1), Ur(l, 1, 0));
  }
  function jn(l) {
    for (; l === Lf; )
      Lf = hi[--yi], hi[yi] = null, vo = hi[--yi], hi[yi] = null;
    for (; l === Du; )
      Du = Xa[--fa], Xa[fa] = null, Qt = Xa[--fa], Xa[fa] = null, mn = Xa[--fa], Xa[fa] = null;
  }
  var Pt = null, dt = null, rt = !1, Qa = null, Za = !1, rc = Error(_(519));
  function zu(l) {
    var n = Error(_(418, ""));
    throw So(Oa(n, l)), rc;
  }
  function Vf(l) {
    var n = l.stateNode, u = l.type, c = l.memoizedProps;
    switch (n[vl] = l, n[kl] = c, u) {
      case "dialog":
        Le("cancel", n), Le("close", n);
        break;
      case "iframe":
      case "object":
      case "embed":
        Le("load", n);
        break;
      case "video":
      case "audio":
        for (u = 0; u < zs.length; u++)
          Le(zs[u], n);
        break;
      case "source":
        Le("error", n);
        break;
      case "img":
      case "image":
      case "link":
        Le("error", n), Le("load", n);
        break;
      case "details":
        Le("toggle", n);
        break;
      case "input":
        Le("invalid", n), yr(
          n,
          c.value,
          c.defaultValue,
          c.checked,
          c.defaultChecked,
          c.type,
          c.name,
          !0
        ), ai(n);
        break;
      case "select":
        Le("invalid", n);
        break;
      case "textarea":
        Le("invalid", n), Bh(n, c.value, c.defaultValue, c.children), ai(n);
    }
    u = c.children, typeof u != "string" && typeof u != "number" && typeof u != "bigint" || n.textContent === "" + u || c.suppressHydrationWarning === !0 || Am(n.textContent, u) ? (c.popover != null && (Le("beforetoggle", n), Le("toggle", n)), c.onScroll != null && Le("scroll", n), c.onScrollEnd != null && Le("scrollend", n), c.onClick != null && (n.onclick = Gd), n = !0) : n = !1, n || zu(l);
  }
  function sy(l) {
    for (Pt = l.return; Pt; )
      switch (Pt.tag) {
        case 5:
        case 13:
          Za = !1;
          return;
        case 27:
        case 3:
          Za = !0;
          return;
        default:
          Pt = Pt.return;
      }
  }
  function go(l) {
    if (l !== Pt) return !1;
    if (!rt) return sy(l), rt = !0, !1;
    var n = l.tag, u;
    if ((u = n !== 3 && n !== 27) && ((u = n === 5) && (u = l.type, u = !(u !== "form" && u !== "button") || au(l.type, l.memoizedProps)), u = !u), u && dt && zu(l), sy(l), n === 13) {
      if (l = l.memoizedState, l = l !== null ? l.dehydrated : null, !l) throw Error(_(317));
      e: {
        for (l = l.nextSibling, n = 0; l; ) {
          if (l.nodeType === 8)
            if (u = l.data, u === "/$") {
              if (n === 0) {
                dt = Tn(l.nextSibling);
                break e;
              }
              n--;
            } else
              u !== "$" && u !== "$!" && u !== "$?" || n++;
          l = l.nextSibling;
        }
        dt = null;
      }
    } else
      n === 27 ? (n = dt, xi(l.type) ? (l = Hi, Hi = null, dt = l) : dt = n) : dt = Pt ? Tn(l.stateNode.nextSibling) : null;
    return !0;
  }
  function bo() {
    dt = Pt = null, rt = !1;
  }
  function ry() {
    var l = Qa;
    return l !== null && (ma === null ? ma = l : ma.push.apply(
      ma,
      l
    ), Qa = null), l;
  }
  function So(l) {
    Qa === null ? Qa = [l] : Qa.push(l);
  }
  var Xf = j(null), Mu = null, pn = null;
  function _u(l, n, u) {
    I(Xf, n._currentValue), n._currentValue = u;
  }
  function Bn(l) {
    l._currentValue = Xf.current, J(Xf);
  }
  function Cr(l, n, u) {
    for (; l !== null; ) {
      var c = l.alternate;
      if ((l.childLanes & n) !== n ? (l.childLanes |= n, c !== null && (c.childLanes |= n)) : c !== null && (c.childLanes & n) !== n && (c.childLanes |= n), l === u) break;
      l = l.return;
    }
  }
  function dy(l, n, u, c) {
    var s = l.child;
    for (s !== null && (s.return = l); s !== null; ) {
      var r = s.dependencies;
      if (r !== null) {
        var y = s.child;
        r = r.firstContext;
        e: for (; r !== null; ) {
          var p = r;
          r = s;
          for (var S = 0; S < n.length; S++)
            if (p.context === n[S]) {
              r.lanes |= u, p = r.alternate, p !== null && (p.lanes |= u), Cr(
                r.return,
                u,
                l
              ), c || (y = null);
              break e;
            }
          r = p.next;
        }
      } else if (s.tag === 18) {
        if (y = s.return, y === null) throw Error(_(341));
        y.lanes |= u, r = y.alternate, r !== null && (r.lanes |= u), Cr(y, u, l), y = null;
      } else y = s.child;
      if (y !== null) y.return = s;
      else
        for (y = s; y !== null; ) {
          if (y === l) {
            y = null;
            break;
          }
          if (s = y.sibling, s !== null) {
            s.return = y.return, y = s;
            break;
          }
          y = y.return;
        }
      s = y;
    }
  }
  function To(l, n, u, c) {
    l = null;
    for (var s = n, r = !1; s !== null; ) {
      if (!r) {
        if ((s.flags & 524288) !== 0) r = !0;
        else if ((s.flags & 262144) !== 0) break;
      }
      if (s.tag === 10) {
        var y = s.alternate;
        if (y === null) throw Error(_(387));
        if (y = y.memoizedProps, y !== null) {
          var p = s.type;
          Ul(s.pendingProps.value, y.value) || (l !== null ? l.push(p) : l = [p]);
        }
      } else if (s === il.current) {
        if (y = s.alternate, y === null) throw Error(_(387));
        y.memoizedState.memoizedState !== s.memoizedState.memoizedState && (l !== null ? l.push(ba) : l = [ba]);
      }
      s = s.return;
    }
    l !== null && dy(
      n,
      l,
      u,
      c
    ), n.flags |= 262144;
  }
  function Qf(l) {
    for (l = l.firstContext; l !== null; ) {
      if (!Ul(
        l.context._currentValue,
        l.memoizedValue
      ))
        return !0;
      l = l.next;
    }
    return !1;
  }
  function mi(l) {
    Mu = l, pn = null, l = l.dependencies, l !== null && (l.firstContext = null);
  }
  function gl(l) {
    return hy(Mu, l);
  }
  function Zf(l, n) {
    return Mu === null && mi(l), hy(l, n);
  }
  function hy(l, n) {
    var u = n._currentValue;
    if (n = { context: n, memoizedValue: u, next: null }, pn === null) {
      if (l === null) throw Error(_(308));
      pn = n, l.dependencies = { lanes: 0, firstContext: n }, l.flags |= 524288;
    } else pn = pn.next = n;
    return u;
  }
  var Eo = typeof AbortController < "u" ? AbortController : function() {
    var l = [], n = this.signal = {
      aborted: !1,
      addEventListener: function(u, c) {
        l.push(c);
      }
    };
    this.abort = function() {
      n.aborted = !0, l.forEach(function(u) {
        return u();
      });
    };
  }, xr = U.unstable_scheduleCallback, kp = U.unstable_NormalPriority, fl = {
    $$typeof: it,
    Consumer: null,
    Provider: null,
    _currentValue: null,
    _currentValue2: null,
    _threadCount: 0
  };
  function Ao() {
    return {
      controller: new Eo(),
      data: /* @__PURE__ */ new Map(),
      refCount: 0
    };
  }
  function Yn(l) {
    l.refCount--, l.refCount === 0 && xr(kp, function() {
      l.controller.abort();
    });
  }
  var pi = null, Kf = 0, Ka = 0, sl = null;
  function Hr(l, n) {
    if (pi === null) {
      var u = pi = [];
      Kf = 0, Ka = Hc(), sl = {
        status: "pending",
        value: void 0,
        then: function(c) {
          u.push(c);
        }
      };
    }
    return Kf++, n.then(Nr, Nr), n;
  }
  function Nr() {
    if (--Kf === 0 && pi !== null) {
      sl !== null && (sl.status = "fulfilled");
      var l = pi;
      pi = null, Ka = 0, sl = null;
      for (var n = 0; n < l.length; n++) (0, l[n])();
    }
  }
  function $p(l, n) {
    var u = [], c = {
      status: "pending",
      value: null,
      reason: null,
      then: function(s) {
        u.push(s);
      }
    };
    return l.then(
      function() {
        c.status = "fulfilled", c.value = n;
        for (var s = 0; s < u.length; s++) (0, u[s])(n);
      },
      function(s) {
        for (c.status = "rejected", c.reason = s, s = 0; s < u.length; s++)
          (0, u[s])(void 0);
      }
    ), c;
  }
  var wr = O.S;
  O.S = function(l, n) {
    typeof n == "object" && n !== null && typeof n.then == "function" && Hr(l, n), wr !== null && wr(l, n);
  };
  var Gn = j(null);
  function Jf() {
    var l = Gn.current;
    return l !== null ? l : _t.pooledCache;
  }
  function dc(l, n) {
    n === null ? I(Gn, Gn.current) : I(Gn, n.pool);
  }
  function qr() {
    var l = Jf();
    return l === null ? null : { parent: fl._currentValue, pool: l };
  }
  var vi = Error(_(460)), jr = Error(_(474)), kf = Error(_(542)), Br = { then: function() {
  } };
  function Yr(l) {
    return l = l.status, l === "fulfilled" || l === "rejected";
  }
  function $f() {
  }
  function yy(l, n, u) {
    switch (u = l[u], u === void 0 ? l.push(n) : u !== n && (n.then($f, $f), n = u), n.status) {
      case "fulfilled":
        return n.value;
      case "rejected":
        throw l = n.reason, py(l), l;
      default:
        if (typeof n.status == "string") n.then($f, $f);
        else {
          if (l = _t, l !== null && 100 < l.shellSuspendCounter)
            throw Error(_(482));
          l = n, l.status = "pending", l.then(
            function(c) {
              if (n.status === "pending") {
                var s = n;
                s.status = "fulfilled", s.value = c;
              }
            },
            function(c) {
              if (n.status === "pending") {
                var s = n;
                s.status = "rejected", s.reason = c;
              }
            }
          );
        }
        switch (n.status) {
          case "fulfilled":
            return n.value;
          case "rejected":
            throw l = n.reason, py(l), l;
        }
        throw hc = n, vi;
    }
  }
  var hc = null;
  function my() {
    if (hc === null) throw Error(_(459));
    var l = hc;
    return hc = null, l;
  }
  function py(l) {
    if (l === vi || l === kf)
      throw Error(_(483));
  }
  var Ln = !1;
  function Gr(l) {
    l.updateQueue = {
      baseState: l.memoizedState,
      firstBaseUpdate: null,
      lastBaseUpdate: null,
      shared: { pending: null, lanes: 0, hiddenCallbacks: null },
      callbacks: null
    };
  }
  function Lr(l, n) {
    l = l.updateQueue, n.updateQueue === l && (n.updateQueue = {
      baseState: l.baseState,
      firstBaseUpdate: l.firstBaseUpdate,
      lastBaseUpdate: l.lastBaseUpdate,
      shared: l.shared,
      callbacks: null
    });
  }
  function sa(l) {
    return { lane: l, tag: 0, payload: null, callback: null, next: null };
  }
  function Vn(l, n, u) {
    var c = l.updateQueue;
    if (c === null) return null;
    if (c = c.shared, (gt & 2) !== 0) {
      var s = c.pending;
      return s === null ? n.next = n : (n.next = s.next, s.next = n), c.pending = n, n = Yf(l), yo(l, null, u), n;
    }
    return di(l, c, n, u), Yf(l);
  }
  function yc(l, n, u) {
    if (n = n.updateQueue, n !== null && (n = n.shared, (u & 4194048) !== 0)) {
      var c = n.lanes;
      c &= l.pendingLanes, u |= c, n.lanes = u, Be(l, u);
    }
  }
  function vy(l, n) {
    var u = l.updateQueue, c = l.alternate;
    if (c !== null && (c = c.updateQueue, u === c)) {
      var s = null, r = null;
      if (u = u.firstBaseUpdate, u !== null) {
        do {
          var y = {
            lane: u.lane,
            tag: u.tag,
            payload: u.payload,
            callback: null,
            next: null
          };
          r === null ? s = r = y : r = r.next = y, u = u.next;
        } while (u !== null);
        r === null ? s = r = n : r = r.next = n;
      } else s = r = n;
      u = {
        baseState: c.baseState,
        firstBaseUpdate: s,
        lastBaseUpdate: r,
        shared: c.shared,
        callbacks: c.callbacks
      }, l.updateQueue = u;
      return;
    }
    l = u.lastBaseUpdate, l === null ? u.firstBaseUpdate = n : l.next = n, u.lastBaseUpdate = n;
  }
  var gy = !1;
  function Ro() {
    if (gy) {
      var l = sl;
      if (l !== null) throw l;
    }
  }
  function Uu(l, n, u, c) {
    gy = !1;
    var s = l.updateQueue;
    Ln = !1;
    var r = s.firstBaseUpdate, y = s.lastBaseUpdate, p = s.shared.pending;
    if (p !== null) {
      s.shared.pending = null;
      var S = p, H = S.next;
      S.next = null, y === null ? r = H : y.next = H, y = S;
      var K = l.alternate;
      K !== null && (K = K.updateQueue, p = K.lastBaseUpdate, p !== y && (p === null ? K.firstBaseUpdate = H : p.next = H, K.lastBaseUpdate = S));
    }
    if (r !== null) {
      var $ = s.baseState;
      y = 0, K = H = S = null, p = r;
      do {
        var q = p.lane & -536870913, Y = q !== p.lane;
        if (Y ? (lt & q) === q : (c & q) === q) {
          q !== 0 && q === Ka && (gy = !0), K !== null && (K = K.next = {
            lane: 0,
            tag: p.tag,
            payload: p.payload,
            callback: null,
            next: null
          });
          e: {
            var Ae = l, Re = p;
            q = n;
            var yt = u;
            switch (Re.tag) {
              case 1:
                if (Ae = Re.payload, typeof Ae == "function") {
                  $ = Ae.call(yt, $, q);
                  break e;
                }
                $ = Ae;
                break e;
              case 3:
                Ae.flags = Ae.flags & -65537 | 128;
              case 0:
                if (Ae = Re.payload, q = typeof Ae == "function" ? Ae.call(yt, $, q) : Ae, q == null) break e;
                $ = te({}, $, q);
                break e;
              case 2:
                Ln = !0;
            }
          }
          q = p.callback, q !== null && (l.flags |= 64, Y && (l.flags |= 8192), Y = s.callbacks, Y === null ? s.callbacks = [q] : Y.push(q));
        } else
          Y = {
            lane: q,
            tag: p.tag,
            payload: p.payload,
            callback: p.callback,
            next: null
          }, K === null ? (H = K = Y, S = $) : K = K.next = Y, y |= q;
        if (p = p.next, p === null) {
          if (p = s.shared.pending, p === null)
            break;
          Y = p, p = Y.next, Y.next = null, s.lastBaseUpdate = Y, s.shared.pending = null;
        }
      } while (!0);
      K === null && (S = $), s.baseState = S, s.firstBaseUpdate = H, s.lastBaseUpdate = K, r === null && (s.shared.lanes = 0), ju |= y, l.lanes = y, l.memoizedState = $;
    }
  }
  function Vr(l, n) {
    if (typeof l != "function")
      throw Error(_(191, l));
    l.call(n);
  }
  function Wf(l, n) {
    var u = l.callbacks;
    if (u !== null)
      for (l.callbacks = null, l = 0; l < u.length; l++)
        Vr(u[l], n);
  }
  var mc = j(null), Ff = j(0);
  function bl(l, n) {
    l = qu, I(Ff, l), I(mc, n), qu = l | n.baseLanes;
  }
  function Oo() {
    I(Ff, qu), I(mc, mc.current);
  }
  function Do() {
    qu = Ff.current, J(mc), J(Ff);
  }
  var Ja = 0, Ge = null, vt = null, Vt = null, If = !1, Da = !1, gi = !1, vn = 0, za = 0, Cu = null, by = 0;
  function Xt() {
    throw Error(_(321));
  }
  function Xr(l, n) {
    if (n === null) return !1;
    for (var u = 0; u < n.length && u < l.length; u++)
      if (!Ul(l[u], n[u])) return !1;
    return !0;
  }
  function Qr(l, n, u, c, s, r) {
    return Ja = r, Ge = n, n.memoizedState = null, n.updateQueue = null, n.lanes = 0, O.H = l === null || l.memoizedState === null ? Ny : wy, gi = !1, r = u(c, s), gi = !1, Da && (r = Sy(
      n,
      u,
      c,
      s
    )), bi(l), r;
  }
  function bi(l) {
    O.H = cd;
    var n = vt !== null && vt.next !== null;
    if (Ja = 0, Vt = vt = Ge = null, If = !1, za = 0, Cu = null, n) throw Error(_(300));
    l === null || rl || (l = l.dependencies, l !== null && Qf(l) && (rl = !0));
  }
  function Sy(l, n, u, c) {
    Ge = l;
    var s = 0;
    do {
      if (Da && (Cu = null), za = 0, Da = !1, 25 <= s) throw Error(_(301));
      if (s += 1, Vt = vt = null, l.updateQueue != null) {
        var r = l.updateQueue;
        r.lastEffect = null, r.events = null, r.stores = null, r.memoCache != null && (r.memoCache.index = 0);
      }
      O.H = xu, r = n(u, c);
    } while (Da);
    return r;
  }
  function Wp() {
    var l = O.H, n = l.useState()[0];
    return n = typeof n.then == "function" ? es(n) : n, l = l.useState()[0], (vt !== null ? vt.memoizedState : null) !== l && (Ge.flags |= 1024), n;
  }
  function Zr() {
    var l = vn !== 0;
    return vn = 0, l;
  }
  function zo(l, n, u) {
    n.updateQueue = l.updateQueue, n.flags &= -2053, l.lanes &= ~u;
  }
  function Kr(l) {
    if (If) {
      for (l = l.memoizedState; l !== null; ) {
        var n = l.queue;
        n !== null && (n.pending = null), l = l.next;
      }
      If = !1;
    }
    Ja = 0, Vt = vt = Ge = null, Da = !1, za = vn = 0, Cu = null;
  }
  function Ll() {
    var l = {
      memoizedState: null,
      baseState: null,
      baseQueue: null,
      queue: null,
      next: null
    };
    return Vt === null ? Ge.memoizedState = Vt = l : Vt = Vt.next = l, Vt;
  }
  function Zt() {
    if (vt === null) {
      var l = Ge.alternate;
      l = l !== null ? l.memoizedState : null;
    } else l = vt.next;
    var n = Vt === null ? Ge.memoizedState : Vt.next;
    if (n !== null)
      Vt = n, vt = l;
    else {
      if (l === null)
        throw Ge.alternate === null ? Error(_(467)) : Error(_(310));
      vt = l, l = {
        memoizedState: vt.memoizedState,
        baseState: vt.baseState,
        baseQueue: vt.baseQueue,
        queue: vt.queue,
        next: null
      }, Vt === null ? Ge.memoizedState = Vt = l : Vt = Vt.next = l;
    }
    return Vt;
  }
  function Pf() {
    return { lastEffect: null, events: null, stores: null, memoCache: null };
  }
  function es(l) {
    var n = za;
    return za += 1, Cu === null && (Cu = []), l = yy(Cu, l, n), n = Ge, (Vt === null ? n.memoizedState : Vt.next) === null && (n = n.alternate, O.H = n === null || n.memoizedState === null ? Ny : wy), l;
  }
  function al(l) {
    if (l !== null && typeof l == "object") {
      if (typeof l.then == "function") return es(l);
      if (l.$$typeof === it) return gl(l);
    }
    throw Error(_(438, String(l)));
  }
  function Jr(l) {
    var n = null, u = Ge.updateQueue;
    if (u !== null && (n = u.memoCache), n == null) {
      var c = Ge.alternate;
      c !== null && (c = c.updateQueue, c !== null && (c = c.memoCache, c != null && (n = {
        data: c.data.map(function(s) {
          return s.slice();
        }),
        index: 0
      })));
    }
    if (n == null && (n = { data: [], index: 0 }), u === null && (u = Pf(), Ge.updateQueue = u), u.memoCache = n, u = n.data[n.index], u === void 0)
      for (u = n.data[n.index] = Array(l), c = 0; c < l; c++)
        u[c] = Rt;
    return n.index++, u;
  }
  function Xn(l, n) {
    return typeof n == "function" ? n(l) : n;
  }
  function ts(l) {
    var n = Zt();
    return kr(n, vt, l);
  }
  function kr(l, n, u) {
    var c = l.queue;
    if (c === null) throw Error(_(311));
    c.lastRenderedReducer = u;
    var s = l.baseQueue, r = c.pending;
    if (r !== null) {
      if (s !== null) {
        var y = s.next;
        s.next = r.next, r.next = y;
      }
      n.baseQueue = s = r, c.pending = null;
    }
    if (r = l.baseState, s === null) l.memoizedState = r;
    else {
      n = s.next;
      var p = y = null, S = null, H = n, K = !1;
      do {
        var $ = H.lane & -536870913;
        if ($ !== H.lane ? (lt & $) === $ : (Ja & $) === $) {
          var q = H.revertLane;
          if (q === 0)
            S !== null && (S = S.next = {
              lane: 0,
              revertLane: 0,
              action: H.action,
              hasEagerState: H.hasEagerState,
              eagerState: H.eagerState,
              next: null
            }), $ === Ka && (K = !0);
          else if ((Ja & q) === q) {
            H = H.next, q === Ka && (K = !0);
            continue;
          } else
            $ = {
              lane: 0,
              revertLane: H.revertLane,
              action: H.action,
              hasEagerState: H.hasEagerState,
              eagerState: H.eagerState,
              next: null
            }, S === null ? (p = S = $, y = r) : S = S.next = $, Ge.lanes |= q, ju |= q;
          $ = H.action, gi && u(r, $), r = H.hasEagerState ? H.eagerState : u(r, $);
        } else
          q = {
            lane: $,
            revertLane: H.revertLane,
            action: H.action,
            hasEagerState: H.hasEagerState,
            eagerState: H.eagerState,
            next: null
          }, S === null ? (p = S = q, y = r) : S = S.next = q, Ge.lanes |= $, ju |= $;
        H = H.next;
      } while (H !== null && H !== n);
      if (S === null ? y = r : S.next = p, !Ul(r, l.memoizedState) && (rl = !0, K && (u = sl, u !== null)))
        throw u;
      l.memoizedState = r, l.baseState = y, l.baseQueue = S, c.lastRenderedState = r;
    }
    return s === null && (c.lanes = 0), [l.memoizedState, c.dispatch];
  }
  function $r(l) {
    var n = Zt(), u = n.queue;
    if (u === null) throw Error(_(311));
    u.lastRenderedReducer = l;
    var c = u.dispatch, s = u.pending, r = n.memoizedState;
    if (s !== null) {
      u.pending = null;
      var y = s = s.next;
      do
        r = l(r, y.action), y = y.next;
      while (y !== s);
      Ul(r, n.memoizedState) || (rl = !0), n.memoizedState = r, n.baseQueue === null && (n.baseState = r), u.lastRenderedState = r;
    }
    return [r, c];
  }
  function ls(l, n, u) {
    var c = Ge, s = Zt(), r = rt;
    if (r) {
      if (u === void 0) throw Error(_(407));
      u = u();
    } else u = n();
    var y = !Ul(
      (vt || s).memoizedState,
      u
    );
    y && (s.memoizedState = u, rl = !0), s = s.queue;
    var p = Ey.bind(null, c, s, l);
    if (Dt(2048, 8, p, [l]), s.getSnapshot !== n || y || Vt !== null && Vt.memoizedState.tag & 1) {
      if (c.flags |= 2048, ra(
        9,
        us(),
        Ty.bind(
          null,
          c,
          s,
          u,
          n
        ),
        null
      ), _t === null) throw Error(_(349));
      r || (Ja & 124) !== 0 || Wr(c, n, u);
    }
    return u;
  }
  function Wr(l, n, u) {
    l.flags |= 16384, l = { getSnapshot: n, value: u }, n = Ge.updateQueue, n === null ? (n = Pf(), Ge.updateQueue = n, n.stores = [l]) : (u = n.stores, u === null ? n.stores = [l] : u.push(l));
  }
  function Ty(l, n, u, c) {
    n.value = u, n.getSnapshot = c, Ay(n) && Fr(l);
  }
  function Ey(l, n, u) {
    return u(function() {
      Ay(n) && Fr(l);
    });
  }
  function Ay(l) {
    var n = l.getSnapshot;
    l = l.value;
    try {
      var u = n();
      return !Ul(l, u);
    } catch {
      return !0;
    }
  }
  function Fr(l) {
    var n = qn(l, 2);
    n !== null && Ua(n, l, 2);
  }
  function as(l) {
    var n = Ll();
    if (typeof l == "function") {
      var u = l;
      if (l = u(), gi) {
        Ba(!0);
        try {
          u();
        } finally {
          Ba(!1);
        }
      }
    }
    return n.memoizedState = n.baseState = l, n.queue = {
      pending: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: Xn,
      lastRenderedState: l
    }, n;
  }
  function Ir(l, n, u, c) {
    return l.baseState = u, kr(
      l,
      vt,
      typeof c == "function" ? c : Xn
    );
  }
  function Fp(l, n, u, c, s) {
    if (bc(l)) throw Error(_(485));
    if (l = n.action, l !== null) {
      var r = {
        payload: s,
        action: l,
        next: null,
        isTransition: !0,
        status: "pending",
        value: null,
        reason: null,
        listeners: [],
        then: function(y) {
          r.listeners.push(y);
        }
      };
      O.T !== null ? u(!0) : r.isTransition = !1, c(r), u = n.pending, u === null ? (r.next = n.pending = r, Pr(n, r)) : (r.next = u.next, n.pending = u.next = r);
    }
  }
  function Pr(l, n) {
    var u = n.action, c = n.payload, s = l.state;
    if (n.isTransition) {
      var r = O.T, y = {};
      O.T = y;
      try {
        var p = u(s, c), S = O.S;
        S !== null && S(y, p), ns(l, n, p);
      } catch (H) {
        td(l, n, H);
      } finally {
        O.T = r;
      }
    } else
      try {
        r = u(s, c), ns(l, n, r);
      } catch (H) {
        td(l, n, H);
      }
  }
  function ns(l, n, u) {
    u !== null && typeof u == "object" && typeof u.then == "function" ? u.then(
      function(c) {
        ed(l, n, c);
      },
      function(c) {
        return td(l, n, c);
      }
    ) : ed(l, n, u);
  }
  function ed(l, n, u) {
    n.status = "fulfilled", n.value = u, Ry(n), l.state = u, n = l.pending, n !== null && (u = n.next, u === n ? l.pending = null : (u = u.next, n.next = u, Pr(l, u)));
  }
  function td(l, n, u) {
    var c = l.pending;
    if (l.pending = null, c !== null) {
      c = c.next;
      do
        n.status = "rejected", n.reason = u, Ry(n), n = n.next;
      while (n !== c);
    }
    l.action = null;
  }
  function Ry(l) {
    l = l.listeners;
    for (var n = 0; n < l.length; n++) (0, l[n])();
  }
  function ld(l, n) {
    return n;
  }
  function Oy(l, n) {
    if (rt) {
      var u = _t.formState;
      if (u !== null) {
        e: {
          var c = Ge;
          if (rt) {
            if (dt) {
              t: {
                for (var s = dt, r = Za; s.nodeType !== 8; ) {
                  if (!r) {
                    s = null;
                    break t;
                  }
                  if (s = Tn(
                    s.nextSibling
                  ), s === null) {
                    s = null;
                    break t;
                  }
                }
                r = s.data, s = r === "F!" || r === "F" ? s : null;
              }
              if (s) {
                dt = Tn(
                  s.nextSibling
                ), c = s.data === "F!";
                break e;
              }
            }
            zu(c);
          }
          c = !1;
        }
        c && (n = u[0]);
      }
    }
    return u = Ll(), u.memoizedState = u.baseState = n, c = {
      pending: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: ld,
      lastRenderedState: n
    }, u.queue = c, u = xy.bind(
      null,
      Ge,
      c
    ), c.dispatch = u, c = as(!1), r = os.bind(
      null,
      Ge,
      !1,
      c.queue
    ), c = Ll(), s = {
      state: n,
      dispatch: null,
      action: l,
      pending: null
    }, c.queue = s, u = Fp.bind(
      null,
      Ge,
      s,
      r,
      u
    ), s.dispatch = u, c.memoizedState = l, [n, u, !1];
  }
  function Qn(l) {
    var n = Zt();
    return ad(n, vt, l);
  }
  function ad(l, n, u) {
    if (n = kr(
      l,
      n,
      ld
    )[0], l = ts(Xn)[0], typeof n == "object" && n !== null && typeof n.then == "function")
      try {
        var c = es(n);
      } catch (y) {
        throw y === vi ? kf : y;
      }
    else c = n;
    n = Zt();
    var s = n.queue, r = s.dispatch;
    return u !== n.memoizedState && (Ge.flags |= 2048, ra(
      9,
      us(),
      Mg.bind(null, s, u),
      null
    )), [c, r, l];
  }
  function Mg(l, n) {
    l.action = n;
  }
  function nd(l) {
    var n = Zt(), u = vt;
    if (u !== null)
      return ad(n, u, l);
    Zt(), n = n.memoizedState, u = Zt();
    var c = u.queue.dispatch;
    return u.memoizedState = l, [n, c, !1];
  }
  function ra(l, n, u, c) {
    return l = { tag: l, create: u, deps: c, inst: n, next: null }, n = Ge.updateQueue, n === null && (n = Pf(), Ge.updateQueue = n), u = n.lastEffect, u === null ? n.lastEffect = l.next = l : (c = u.next, u.next = l, l.next = c, n.lastEffect = l), l;
  }
  function us() {
    return { destroy: void 0, resource: void 0 };
  }
  function is() {
    return Zt().memoizedState;
  }
  function Si(l, n, u, c) {
    var s = Ll();
    c = c === void 0 ? null : c, Ge.flags |= l, s.memoizedState = ra(
      1 | n,
      us(),
      u,
      c
    );
  }
  function Dt(l, n, u, c) {
    var s = Zt();
    c = c === void 0 ? null : c;
    var r = s.memoizedState.inst;
    vt !== null && c !== null && Xr(c, vt.memoizedState.deps) ? s.memoizedState = ra(n, r, u, c) : (Ge.flags |= l, s.memoizedState = ra(
      1 | n,
      r,
      u,
      c
    ));
  }
  function Ip(l, n) {
    Si(8390656, 8, l, n);
  }
  function Pp(l, n) {
    Dt(2048, 8, l, n);
  }
  function Dy(l, n) {
    return Dt(4, 2, l, n);
  }
  function gn(l, n) {
    return Dt(4, 4, l, n);
  }
  function zy(l, n) {
    if (typeof n == "function") {
      l = l();
      var u = n(l);
      return function() {
        typeof u == "function" ? u() : n(null);
      };
    }
    if (n != null)
      return l = l(), n.current = l, function() {
        n.current = null;
      };
  }
  function ud(l, n, u) {
    u = u != null ? u.concat([l]) : null, Dt(4, 4, zy.bind(null, n, l), u);
  }
  function pc() {
  }
  function vc(l, n) {
    var u = Zt();
    n = n === void 0 ? null : n;
    var c = u.memoizedState;
    return n !== null && Xr(n, c[1]) ? c[0] : (u.memoizedState = [l, n], l);
  }
  function My(l, n) {
    var u = Zt();
    n = n === void 0 ? null : n;
    var c = u.memoizedState;
    if (n !== null && Xr(n, c[1]))
      return c[0];
    if (c = l(), gi) {
      Ba(!0);
      try {
        l();
      } finally {
        Ba(!1);
      }
    }
    return u.memoizedState = [c, n], c;
  }
  function cs(l, n, u) {
    return u === void 0 || (Ja & 1073741824) !== 0 ? l.memoizedState = n : (l.memoizedState = u, l = sm(), Ge.lanes |= l, ju |= l, u);
  }
  function _y(l, n, u, c) {
    return Ul(u, n) ? u : mc.current !== null ? (l = cs(l, u, c), Ul(l, n) || (rl = !0), l) : (Ja & 42) === 0 ? (rl = !0, l.memoizedState = u) : (l = sm(), Ge.lanes |= l, ju |= l, n);
  }
  function ev(l, n, u, c, s) {
    var r = F.p;
    F.p = r !== 0 && 8 > r ? r : 8;
    var y = O.T, p = {};
    O.T = p, os(l, !1, n, u);
    try {
      var S = s(), H = O.S;
      if (H !== null && H(p, S), S !== null && typeof S == "object" && typeof S.then == "function") {
        var K = $p(
          S,
          c
        );
        gc(
          l,
          n,
          K,
          _a(l)
        );
      } else
        gc(
          l,
          n,
          c,
          _a(l)
        );
    } catch ($) {
      gc(
        l,
        n,
        { then: function() {
        }, status: "rejected", reason: $ },
        _a()
      );
    } finally {
      F.p = r, O.T = y;
    }
  }
  function _g() {
  }
  function id(l, n, u, c) {
    if (l.tag !== 5) throw Error(_(476));
    var s = tv(l).queue;
    ev(
      l,
      s,
      n,
      P,
      u === null ? _g : function() {
        return Mo(l), u(c);
      }
    );
  }
  function tv(l) {
    var n = l.memoizedState;
    if (n !== null) return n;
    n = {
      memoizedState: P,
      baseState: P,
      baseQueue: null,
      queue: {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: Xn,
        lastRenderedState: P
      },
      next: null
    };
    var u = {};
    return n.next = {
      memoizedState: u,
      baseState: u,
      baseQueue: null,
      queue: {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: Xn,
        lastRenderedState: u
      },
      next: null
    }, l.memoizedState = n, l = l.alternate, l !== null && (l.memoizedState = n), n;
  }
  function Mo(l) {
    var n = tv(l).next.queue;
    gc(l, n, {}, _a());
  }
  function ka() {
    return gl(ba);
  }
  function Uy() {
    return Zt().memoizedState;
  }
  function lv() {
    return Zt().memoizedState;
  }
  function av(l) {
    for (var n = l.return; n !== null; ) {
      switch (n.tag) {
        case 24:
        case 3:
          var u = _a();
          l = sa(u);
          var c = Vn(n, l, u);
          c !== null && (Ua(c, n, u), yc(c, n, u)), n = { cache: Ao() }, l.payload = n;
          return;
      }
      n = n.return;
    }
  }
  function Cy(l, n, u) {
    var c = _a();
    u = {
      lane: c,
      revertLane: 0,
      action: u,
      hasEagerState: !1,
      eagerState: null,
      next: null
    }, bc(l) ? nv(n, u) : (u = ho(l, n, u, c), u !== null && (Ua(u, l, c), Hy(u, n, c)));
  }
  function xy(l, n, u) {
    var c = _a();
    gc(l, n, u, c);
  }
  function gc(l, n, u, c) {
    var s = {
      lane: c,
      revertLane: 0,
      action: u,
      hasEagerState: !1,
      eagerState: null,
      next: null
    };
    if (bc(l)) nv(n, s);
    else {
      var r = l.alternate;
      if (l.lanes === 0 && (r === null || r.lanes === 0) && (r = n.lastRenderedReducer, r !== null))
        try {
          var y = n.lastRenderedState, p = r(y, u);
          if (s.hasEagerState = !0, s.eagerState = p, Ul(p, y))
            return di(l, n, s, 0), _t === null && hn(), !1;
        } catch {
        } finally {
        }
      if (u = ho(l, n, s, c), u !== null)
        return Ua(u, l, c), Hy(u, n, c), !0;
    }
    return !1;
  }
  function os(l, n, u, c) {
    if (c = {
      lane: 2,
      revertLane: Hc(),
      action: c,
      hasEagerState: !1,
      eagerState: null,
      next: null
    }, bc(l)) {
      if (n) throw Error(_(479));
    } else
      n = ho(
        l,
        u,
        c,
        2
      ), n !== null && Ua(n, l, 2);
  }
  function bc(l) {
    var n = l.alternate;
    return l === Ge || n !== null && n === Ge;
  }
  function nv(l, n) {
    Da = If = !0;
    var u = l.pending;
    u === null ? n.next = n : (n.next = u.next, u.next = n), l.pending = n;
  }
  function Hy(l, n, u) {
    if ((u & 4194048) !== 0) {
      var c = n.lanes;
      c &= l.pendingLanes, u |= c, n.lanes = u, Be(l, u);
    }
  }
  var cd = {
    readContext: gl,
    use: al,
    useCallback: Xt,
    useContext: Xt,
    useEffect: Xt,
    useImperativeHandle: Xt,
    useLayoutEffect: Xt,
    useInsertionEffect: Xt,
    useMemo: Xt,
    useReducer: Xt,
    useRef: Xt,
    useState: Xt,
    useDebugValue: Xt,
    useDeferredValue: Xt,
    useTransition: Xt,
    useSyncExternalStore: Xt,
    useId: Xt,
    useHostTransitionStatus: Xt,
    useFormState: Xt,
    useActionState: Xt,
    useOptimistic: Xt,
    useMemoCache: Xt,
    useCacheRefresh: Xt
  }, Ny = {
    readContext: gl,
    use: al,
    useCallback: function(l, n) {
      return Ll().memoizedState = [
        l,
        n === void 0 ? null : n
      ], l;
    },
    useContext: gl,
    useEffect: Ip,
    useImperativeHandle: function(l, n, u) {
      u = u != null ? u.concat([l]) : null, Si(
        4194308,
        4,
        zy.bind(null, n, l),
        u
      );
    },
    useLayoutEffect: function(l, n) {
      return Si(4194308, 4, l, n);
    },
    useInsertionEffect: function(l, n) {
      Si(4, 2, l, n);
    },
    useMemo: function(l, n) {
      var u = Ll();
      n = n === void 0 ? null : n;
      var c = l();
      if (gi) {
        Ba(!0);
        try {
          l();
        } finally {
          Ba(!1);
        }
      }
      return u.memoizedState = [c, n], c;
    },
    useReducer: function(l, n, u) {
      var c = Ll();
      if (u !== void 0) {
        var s = u(n);
        if (gi) {
          Ba(!0);
          try {
            u(n);
          } finally {
            Ba(!1);
          }
        }
      } else s = n;
      return c.memoizedState = c.baseState = s, l = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: l,
        lastRenderedState: s
      }, c.queue = l, l = l.dispatch = Cy.bind(
        null,
        Ge,
        l
      ), [c.memoizedState, l];
    },
    useRef: function(l) {
      var n = Ll();
      return l = { current: l }, n.memoizedState = l;
    },
    useState: function(l) {
      l = as(l);
      var n = l.queue, u = xy.bind(null, Ge, n);
      return n.dispatch = u, [l.memoizedState, u];
    },
    useDebugValue: pc,
    useDeferredValue: function(l, n) {
      var u = Ll();
      return cs(u, l, n);
    },
    useTransition: function() {
      var l = as(!1);
      return l = ev.bind(
        null,
        Ge,
        l.queue,
        !0,
        !1
      ), Ll().memoizedState = l, [!1, l];
    },
    useSyncExternalStore: function(l, n, u) {
      var c = Ge, s = Ll();
      if (rt) {
        if (u === void 0)
          throw Error(_(407));
        u = u();
      } else {
        if (u = n(), _t === null)
          throw Error(_(349));
        (lt & 124) !== 0 || Wr(c, n, u);
      }
      s.memoizedState = u;
      var r = { value: u, getSnapshot: n };
      return s.queue = r, Ip(Ey.bind(null, c, r, l), [
        l
      ]), c.flags |= 2048, ra(
        9,
        us(),
        Ty.bind(
          null,
          c,
          r,
          u,
          n
        ),
        null
      ), u;
    },
    useId: function() {
      var l = Ll(), n = _t.identifierPrefix;
      if (rt) {
        var u = Qt, c = mn;
        u = (c & ~(1 << 32 - Dl(c) - 1)).toString(32) + u, n = "«" + n + "R" + u, u = vn++, 0 < u && (n += "H" + u.toString(32)), n += "»";
      } else
        u = by++, n = "«" + n + "r" + u.toString(32) + "»";
      return l.memoizedState = n;
    },
    useHostTransitionStatus: ka,
    useFormState: Oy,
    useActionState: Oy,
    useOptimistic: function(l) {
      var n = Ll();
      n.memoizedState = n.baseState = l;
      var u = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: null,
        lastRenderedState: null
      };
      return n.queue = u, n = os.bind(
        null,
        Ge,
        !0,
        u
      ), u.dispatch = n, [l, n];
    },
    useMemoCache: Jr,
    useCacheRefresh: function() {
      return Ll().memoizedState = av.bind(
        null,
        Ge
      );
    }
  }, wy = {
    readContext: gl,
    use: al,
    useCallback: vc,
    useContext: gl,
    useEffect: Pp,
    useImperativeHandle: ud,
    useInsertionEffect: Dy,
    useLayoutEffect: gn,
    useMemo: My,
    useReducer: ts,
    useRef: is,
    useState: function() {
      return ts(Xn);
    },
    useDebugValue: pc,
    useDeferredValue: function(l, n) {
      var u = Zt();
      return _y(
        u,
        vt.memoizedState,
        l,
        n
      );
    },
    useTransition: function() {
      var l = ts(Xn)[0], n = Zt().memoizedState;
      return [
        typeof l == "boolean" ? l : es(l),
        n
      ];
    },
    useSyncExternalStore: ls,
    useId: Uy,
    useHostTransitionStatus: ka,
    useFormState: Qn,
    useActionState: Qn,
    useOptimistic: function(l, n) {
      var u = Zt();
      return Ir(u, vt, l, n);
    },
    useMemoCache: Jr,
    useCacheRefresh: lv
  }, xu = {
    readContext: gl,
    use: al,
    useCallback: vc,
    useContext: gl,
    useEffect: Pp,
    useImperativeHandle: ud,
    useInsertionEffect: Dy,
    useLayoutEffect: gn,
    useMemo: My,
    useReducer: $r,
    useRef: is,
    useState: function() {
      return $r(Xn);
    },
    useDebugValue: pc,
    useDeferredValue: function(l, n) {
      var u = Zt();
      return vt === null ? cs(u, l, n) : _y(
        u,
        vt.memoizedState,
        l,
        n
      );
    },
    useTransition: function() {
      var l = $r(Xn)[0], n = Zt().memoizedState;
      return [
        typeof l == "boolean" ? l : es(l),
        n
      ];
    },
    useSyncExternalStore: ls,
    useId: Uy,
    useHostTransitionStatus: ka,
    useFormState: nd,
    useActionState: nd,
    useOptimistic: function(l, n) {
      var u = Zt();
      return vt !== null ? Ir(u, vt, l, n) : (u.baseState = l, [l, u.queue.dispatch]);
    },
    useMemoCache: Jr,
    useCacheRefresh: lv
  }, Sc = null, _o = 0;
  function od(l) {
    var n = _o;
    return _o += 1, Sc === null && (Sc = []), yy(Sc, l, n);
  }
  function Tc(l, n) {
    n = n.props.ref, l.ref = n !== void 0 ? n : null;
  }
  function Vl(l, n) {
    throw n.$$typeof === G ? Error(_(525)) : (l = Object.prototype.toString.call(n), Error(
      _(
        31,
        l === "[object Object]" ? "object with keys {" + Object.keys(n).join(", ") + "}" : l
      )
    ));
  }
  function qy(l) {
    var n = l._init;
    return n(l._payload);
  }
  function da(l) {
    function n(M, R) {
      if (l) {
        var C = M.deletions;
        C === null ? (M.deletions = [R], M.flags |= 16) : C.push(R);
      }
    }
    function u(M, R) {
      if (!l) return null;
      for (; R !== null; )
        n(M, R), R = R.sibling;
      return null;
    }
    function c(M) {
      for (var R = /* @__PURE__ */ new Map(); M !== null; )
        M.key !== null ? R.set(M.key, M) : R.set(M.index, M), M = M.sibling;
      return R;
    }
    function s(M, R) {
      return M = yn(M, R), M.index = 0, M.sibling = null, M;
    }
    function r(M, R, C) {
      return M.index = C, l ? (C = M.alternate, C !== null ? (C = C.index, C < R ? (M.flags |= 67108866, R) : C) : (M.flags |= 67108866, R)) : (M.flags |= 1048576, R);
    }
    function y(M) {
      return l && M.alternate === null && (M.flags |= 67108866), M;
    }
    function p(M, R, C, k) {
      return R === null || R.tag !== 6 ? (R = po(C, M.mode, k), R.return = M, R) : (R = s(R, C), R.return = M, R);
    }
    function S(M, R, C, k) {
      var de = C.type;
      return de === je ? K(
        M,
        R,
        C.props.children,
        k,
        C.key
      ) : R !== null && (R.elementType === de || typeof de == "object" && de !== null && de.$$typeof === bt && qy(de) === R.type) ? (R = s(R, C.props), Tc(R, C), R.return = M, R) : (R = ee(
        C.type,
        C.key,
        C.props,
        null,
        M.mode,
        k
      ), Tc(R, C), R.return = M, R);
    }
    function H(M, R, C, k) {
      return R === null || R.tag !== 4 || R.stateNode.containerInfo !== C.containerInfo || R.stateNode.implementation !== C.implementation ? (R = Lt(C, M.mode, k), R.return = M, R) : (R = s(R, C.children || []), R.return = M, R);
    }
    function K(M, R, C, k, de) {
      return R === null || R.tag !== 7 ? (R = Va(
        C,
        M.mode,
        k,
        de
      ), R.return = M, R) : (R = s(R, C), R.return = M, R);
    }
    function $(M, R, C) {
      if (typeof R == "string" && R !== "" || typeof R == "number" || typeof R == "bigint")
        return R = po(
          "" + R,
          M.mode,
          C
        ), R.return = M, R;
      if (typeof R == "object" && R !== null) {
        switch (R.$$typeof) {
          case z:
            return C = ee(
              R.type,
              R.key,
              R.props,
              null,
              M.mode,
              C
            ), Tc(C, R), C.return = M, C;
          case ae:
            return R = Lt(
              R,
              M.mode,
              C
            ), R.return = M, R;
          case bt:
            var k = R._init;
            return R = k(R._payload), $(M, R, C);
        }
        if (pt(R) || ze(R))
          return R = Va(
            R,
            M.mode,
            C,
            null
          ), R.return = M, R;
        if (typeof R.then == "function")
          return $(M, od(R), C);
        if (R.$$typeof === it)
          return $(
            M,
            Zf(M, R),
            C
          );
        Vl(M, R);
      }
      return null;
    }
    function q(M, R, C, k) {
      var de = R !== null ? R.key : null;
      if (typeof C == "string" && C !== "" || typeof C == "number" || typeof C == "bigint")
        return de !== null ? null : p(M, R, "" + C, k);
      if (typeof C == "object" && C !== null) {
        switch (C.$$typeof) {
          case z:
            return C.key === de ? S(M, R, C, k) : null;
          case ae:
            return C.key === de ? H(M, R, C, k) : null;
          case bt:
            return de = C._init, C = de(C._payload), q(M, R, C, k);
        }
        if (pt(C) || ze(C))
          return de !== null ? null : K(M, R, C, k, null);
        if (typeof C.then == "function")
          return q(
            M,
            R,
            od(C),
            k
          );
        if (C.$$typeof === it)
          return q(
            M,
            R,
            Zf(M, C),
            k
          );
        Vl(M, C);
      }
      return null;
    }
    function Y(M, R, C, k, de) {
      if (typeof k == "string" && k !== "" || typeof k == "number" || typeof k == "bigint")
        return M = M.get(C) || null, p(R, M, "" + k, de);
      if (typeof k == "object" && k !== null) {
        switch (k.$$typeof) {
          case z:
            return M = M.get(
              k.key === null ? C : k.key
            ) || null, S(R, M, k, de);
          case ae:
            return M = M.get(
              k.key === null ? C : k.key
            ) || null, H(R, M, k, de);
          case bt:
            var We = k._init;
            return k = We(k._payload), Y(
              M,
              R,
              C,
              k,
              de
            );
        }
        if (pt(k) || ze(k))
          return M = M.get(C) || null, K(R, M, k, de, null);
        if (typeof k.then == "function")
          return Y(
            M,
            R,
            C,
            od(k),
            de
          );
        if (k.$$typeof === it)
          return Y(
            M,
            R,
            C,
            Zf(R, k),
            de
          );
        Vl(R, k);
      }
      return null;
    }
    function Ae(M, R, C, k) {
      for (var de = null, We = null, Te = R, _e = R = 0, El = null; Te !== null && _e < C.length; _e++) {
        Te.index > _e ? (El = Te, Te = null) : El = Te.sibling;
        var st = q(
          M,
          Te,
          C[_e],
          k
        );
        if (st === null) {
          Te === null && (Te = El);
          break;
        }
        l && Te && st.alternate === null && n(M, Te), R = r(st, R, _e), We === null ? de = st : We.sibling = st, We = st, Te = El;
      }
      if (_e === C.length)
        return u(M, Te), rt && ot(M, _e), de;
      if (Te === null) {
        for (; _e < C.length; _e++)
          Te = $(M, C[_e], k), Te !== null && (R = r(
            Te,
            R,
            _e
          ), We === null ? de = Te : We.sibling = Te, We = Te);
        return rt && ot(M, _e), de;
      }
      for (Te = c(Te); _e < C.length; _e++)
        El = Y(
          Te,
          M,
          _e,
          C[_e],
          k
        ), El !== null && (l && El.alternate !== null && Te.delete(
          El.key === null ? _e : El.key
        ), R = r(
          El,
          R,
          _e
        ), We === null ? de = El : We.sibling = El, We = El);
      return l && Te.forEach(function(Bi) {
        return n(M, Bi);
      }), rt && ot(M, _e), de;
    }
    function Re(M, R, C, k) {
      if (C == null) throw Error(_(151));
      for (var de = null, We = null, Te = R, _e = R = 0, El = null, st = C.next(); Te !== null && !st.done; _e++, st = C.next()) {
        Te.index > _e ? (El = Te, Te = null) : El = Te.sibling;
        var Bi = q(M, Te, st.value, k);
        if (Bi === null) {
          Te === null && (Te = El);
          break;
        }
        l && Te && Bi.alternate === null && n(M, Te), R = r(Bi, R, _e), We === null ? de = Bi : We.sibling = Bi, We = Bi, Te = El;
      }
      if (st.done)
        return u(M, Te), rt && ot(M, _e), de;
      if (Te === null) {
        for (; !st.done; _e++, st = C.next())
          st = $(M, st.value, k), st !== null && (R = r(st, R, _e), We === null ? de = st : We.sibling = st, We = st);
        return rt && ot(M, _e), de;
      }
      for (Te = c(Te); !st.done; _e++, st = C.next())
        st = Y(Te, M, _e, st.value, k), st !== null && (l && st.alternate !== null && Te.delete(st.key === null ? _e : st.key), R = r(st, R, _e), We === null ? de = st : We.sibling = st, We = st);
      return l && Te.forEach(function(Lg) {
        return n(M, Lg);
      }), rt && ot(M, _e), de;
    }
    function yt(M, R, C, k) {
      if (typeof C == "object" && C !== null && C.type === je && C.key === null && (C = C.props.children), typeof C == "object" && C !== null) {
        switch (C.$$typeof) {
          case z:
            e: {
              for (var de = C.key; R !== null; ) {
                if (R.key === de) {
                  if (de = C.type, de === je) {
                    if (R.tag === 7) {
                      u(
                        M,
                        R.sibling
                      ), k = s(
                        R,
                        C.props.children
                      ), k.return = M, M = k;
                      break e;
                    }
                  } else if (R.elementType === de || typeof de == "object" && de !== null && de.$$typeof === bt && qy(de) === R.type) {
                    u(
                      M,
                      R.sibling
                    ), k = s(R, C.props), Tc(k, C), k.return = M, M = k;
                    break e;
                  }
                  u(M, R);
                  break;
                } else n(M, R);
                R = R.sibling;
              }
              C.type === je ? (k = Va(
                C.props.children,
                M.mode,
                k,
                C.key
              ), k.return = M, M = k) : (k = ee(
                C.type,
                C.key,
                C.props,
                null,
                M.mode,
                k
              ), Tc(k, C), k.return = M, M = k);
            }
            return y(M);
          case ae:
            e: {
              for (de = C.key; R !== null; ) {
                if (R.key === de)
                  if (R.tag === 4 && R.stateNode.containerInfo === C.containerInfo && R.stateNode.implementation === C.implementation) {
                    u(
                      M,
                      R.sibling
                    ), k = s(R, C.children || []), k.return = M, M = k;
                    break e;
                  } else {
                    u(M, R);
                    break;
                  }
                else n(M, R);
                R = R.sibling;
              }
              k = Lt(C, M.mode, k), k.return = M, M = k;
            }
            return y(M);
          case bt:
            return de = C._init, C = de(C._payload), yt(
              M,
              R,
              C,
              k
            );
        }
        if (pt(C))
          return Ae(
            M,
            R,
            C,
            k
          );
        if (ze(C)) {
          if (de = ze(C), typeof de != "function") throw Error(_(150));
          return C = de.call(C), Re(
            M,
            R,
            C,
            k
          );
        }
        if (typeof C.then == "function")
          return yt(
            M,
            R,
            od(C),
            k
          );
        if (C.$$typeof === it)
          return yt(
            M,
            R,
            Zf(M, C),
            k
          );
        Vl(M, C);
      }
      return typeof C == "string" && C !== "" || typeof C == "number" || typeof C == "bigint" ? (C = "" + C, R !== null && R.tag === 6 ? (u(M, R.sibling), k = s(R, C), k.return = M, M = k) : (u(M, R), k = po(C, M.mode, k), k.return = M, M = k), y(M)) : u(M, R);
    }
    return function(M, R, C, k) {
      try {
        _o = 0;
        var de = yt(
          M,
          R,
          C,
          k
        );
        return Sc = null, de;
      } catch (Te) {
        if (Te === vi || Te === kf) throw Te;
        var We = oa(29, Te, null, M.mode);
        return We.lanes = k, We.return = M, We;
      } finally {
      }
    };
  }
  var Ec = da(!0), Zn = da(!1), Ma = j(null), Xl = null;
  function Hu(l) {
    var n = l.alternate;
    I(zt, zt.current & 1), I(Ma, l), Xl === null && (n === null || mc.current !== null || n.memoizedState !== null) && (Xl = l);
  }
  function Kn(l) {
    if (l.tag === 22) {
      if (I(zt, zt.current), I(Ma, l), Xl === null) {
        var n = l.alternate;
        n !== null && n.memoizedState !== null && (Xl = l);
      }
    } else Jn();
  }
  function Jn() {
    I(zt, zt.current), I(Ma, Ma.current);
  }
  function bn(l) {
    J(Ma), Xl === l && (Xl = null), J(zt);
  }
  var zt = j(0);
  function fs(l) {
    for (var n = l; n !== null; ) {
      if (n.tag === 13) {
        var u = n.memoizedState;
        if (u !== null && (u = u.dehydrated, u === null || u.data === "$?" || xs(u)))
          return n;
      } else if (n.tag === 19 && n.memoizedProps.revealOrder !== void 0) {
        if ((n.flags & 128) !== 0) return n;
      } else if (n.child !== null) {
        n.child.return = n, n = n.child;
        continue;
      }
      if (n === l) break;
      for (; n.sibling === null; ) {
        if (n.return === null || n.return === l) return null;
        n = n.return;
      }
      n.sibling.return = n.return, n = n.sibling;
    }
    return null;
  }
  function Ti(l, n, u, c) {
    n = l.memoizedState, u = u(c, n), u = u == null ? n : te({}, n, u), l.memoizedState = u, l.lanes === 0 && (l.updateQueue.baseState = u);
  }
  var fd = {
    enqueueSetState: function(l, n, u) {
      l = l._reactInternals;
      var c = _a(), s = sa(c);
      s.payload = n, u != null && (s.callback = u), n = Vn(l, s, c), n !== null && (Ua(n, l, c), yc(n, l, c));
    },
    enqueueReplaceState: function(l, n, u) {
      l = l._reactInternals;
      var c = _a(), s = sa(c);
      s.tag = 1, s.payload = n, u != null && (s.callback = u), n = Vn(l, s, c), n !== null && (Ua(n, l, c), yc(n, l, c));
    },
    enqueueForceUpdate: function(l, n) {
      l = l._reactInternals;
      var u = _a(), c = sa(u);
      c.tag = 2, n != null && (c.callback = n), n = Vn(l, c, u), n !== null && (Ua(n, l, u), yc(n, l, u));
    }
  };
  function Uo(l, n, u, c, s, r, y) {
    return l = l.stateNode, typeof l.shouldComponentUpdate == "function" ? l.shouldComponentUpdate(c, r, y) : n.prototype && n.prototype.isPureReactComponent ? !oi(u, c) || !oi(s, r) : !0;
  }
  function Ac(l, n, u, c) {
    l = n.state, typeof n.componentWillReceiveProps == "function" && n.componentWillReceiveProps(u, c), typeof n.UNSAFE_componentWillReceiveProps == "function" && n.UNSAFE_componentWillReceiveProps(u, c), n.state !== l && fd.enqueueReplaceState(n, n.state, null);
  }
  function Ei(l, n) {
    var u = n;
    if ("ref" in n) {
      u = {};
      for (var c in n)
        c !== "ref" && (u[c] = n[c]);
    }
    if (l = l.defaultProps) {
      u === n && (u = te({}, u));
      for (var s in l)
        u[s] === void 0 && (u[s] = l[s]);
    }
    return u;
  }
  var ss = typeof reportError == "function" ? reportError : function(l) {
    if (typeof window == "object" && typeof window.ErrorEvent == "function") {
      var n = new window.ErrorEvent("error", {
        bubbles: !0,
        cancelable: !0,
        message: typeof l == "object" && l !== null && typeof l.message == "string" ? String(l.message) : String(l),
        error: l
      });
      if (!window.dispatchEvent(n)) return;
    } else if (typeof It == "object" && typeof It.emit == "function") {
      It.emit("uncaughtException", l);
      return;
    }
    console.error(l);
  };
  function Co(l) {
    ss(l);
  }
  function jy(l) {
    console.error(l);
  }
  function rs(l) {
    ss(l);
  }
  function ds(l, n) {
    try {
      var u = l.onUncaughtError;
      u(n.value, { componentStack: n.stack });
    } catch (c) {
      setTimeout(function() {
        throw c;
      });
    }
  }
  function By(l, n, u) {
    try {
      var c = l.onCaughtError;
      c(u.value, {
        componentStack: u.stack,
        errorBoundary: n.tag === 1 ? n.stateNode : null
      });
    } catch (s) {
      setTimeout(function() {
        throw s;
      });
    }
  }
  function Yy(l, n, u) {
    return u = sa(u), u.tag = 3, u.payload = { element: null }, u.callback = function() {
      ds(l, n);
    }, u;
  }
  function Gy(l) {
    return l = sa(l), l.tag = 3, l;
  }
  function ha(l, n, u, c) {
    var s = u.type.getDerivedStateFromError;
    if (typeof s == "function") {
      var r = c.value;
      l.payload = function() {
        return s(r);
      }, l.callback = function() {
        By(n, u, c);
      };
    }
    var y = u.stateNode;
    y !== null && typeof y.componentDidCatch == "function" && (l.callback = function() {
      By(n, u, c), typeof s != "function" && (Di === null ? Di = /* @__PURE__ */ new Set([this]) : Di.add(this));
      var p = c.stack;
      this.componentDidCatch(c.value, {
        componentStack: p !== null ? p : ""
      });
    });
  }
  function uv(l, n, u, c, s) {
    if (u.flags |= 32768, c !== null && typeof c == "object" && typeof c.then == "function") {
      if (n = u.alternate, n !== null && To(
        n,
        u,
        s,
        !0
      ), u = Ma.current, u !== null) {
        switch (u.tag) {
          case 13:
            return Xl === null ? xc() : u.alternate === null && $t === 0 && ($t = 3), u.flags &= -257, u.flags |= 65536, u.lanes = s, c === Br ? u.flags |= 16384 : (n = u.updateQueue, n === null ? u.updateQueue = /* @__PURE__ */ new Set([c]) : n.add(c), wd(l, c, s)), !1;
          case 22:
            return u.flags |= 65536, c === Br ? u.flags |= 16384 : (n = u.updateQueue, n === null ? (n = {
              transitions: null,
              markerInstances: null,
              retryQueue: /* @__PURE__ */ new Set([c])
            }, u.updateQueue = n) : (u = n.retryQueue, u === null ? n.retryQueue = /* @__PURE__ */ new Set([c]) : u.add(c)), wd(l, c, s)), !1;
        }
        throw Error(_(435, u.tag));
      }
      return wd(l, c, s), xc(), !1;
    }
    if (rt)
      return n = Ma.current, n !== null ? ((n.flags & 65536) === 0 && (n.flags |= 256), n.flags |= 65536, n.lanes = s, c !== rc && (l = Error(_(422), { cause: c }), So(Oa(l, u)))) : (c !== rc && (n = Error(_(423), {
        cause: c
      }), So(
        Oa(n, u)
      )), l = l.current.alternate, l.flags |= 65536, s &= -s, l.lanes |= s, c = Oa(c, u), s = Yy(
        l.stateNode,
        c,
        s
      ), vy(l, s), $t !== 4 && ($t = 2)), !1;
    var r = Error(_(520), { cause: c });
    if (r = Oa(r, u), Bo === null ? Bo = [r] : Bo.push(r), $t !== 4 && ($t = 2), n === null) return !0;
    c = Oa(c, u), u = n;
    do {
      switch (u.tag) {
        case 3:
          return u.flags |= 65536, l = s & -s, u.lanes |= l, l = Yy(u.stateNode, c, l), vy(u, l), !1;
        case 1:
          if (n = u.type, r = u.stateNode, (u.flags & 128) === 0 && (typeof n.getDerivedStateFromError == "function" || r !== null && typeof r.componentDidCatch == "function" && (Di === null || !Di.has(r))))
            return u.flags |= 65536, s &= -s, u.lanes |= s, s = Gy(s), ha(
              s,
              l,
              u,
              c
            ), vy(u, s), !1;
      }
      u = u.return;
    } while (u !== null);
    return !1;
  }
  var Kt = Error(_(461)), rl = !1;
  function Sl(l, n, u, c) {
    n.child = l === null ? Zn(n, null, u, c) : Ec(
      n,
      l.child,
      u,
      c
    );
  }
  function iv(l, n, u, c, s) {
    u = u.render;
    var r = n.ref;
    if ("ref" in c) {
      var y = {};
      for (var p in c)
        p !== "ref" && (y[p] = c[p]);
    } else y = c;
    return mi(n), c = Qr(
      l,
      n,
      u,
      y,
      r,
      s
    ), p = Zr(), l !== null && !rl ? (zo(l, n, s), kn(l, n, s)) : (rt && p && sc(n), n.flags |= 1, Sl(l, n, c, s), n.child);
  }
  function Nu(l, n, u, c, s) {
    if (l === null) {
      var r = u.type;
      return typeof r == "function" && !Gf(r) && r.defaultProps === void 0 && u.compare === null ? (n.tag = 15, n.type = r, Rc(
        l,
        n,
        r,
        c,
        s
      )) : (l = ee(
        u.type,
        null,
        c,
        n,
        n.mode,
        s
      ), l.ref = n.ref, l.return = n, n.child = l);
    }
    if (r = l.child, !bd(l, s)) {
      var y = r.memoizedProps;
      if (u = u.compare, u = u !== null ? u : oi, u(y, c) && l.ref === n.ref)
        return kn(l, n, s);
    }
    return n.flags |= 1, l = yn(r, c), l.ref = n.ref, l.return = n, n.child = l;
  }
  function Rc(l, n, u, c, s) {
    if (l !== null) {
      var r = l.memoizedProps;
      if (oi(r, c) && l.ref === n.ref)
        if (rl = !1, n.pendingProps = c = r, bd(l, s))
          (l.flags & 131072) !== 0 && (rl = !0);
        else
          return n.lanes = l.lanes, kn(l, n, s);
    }
    return rd(
      l,
      n,
      u,
      c,
      s
    );
  }
  function sd(l, n, u) {
    var c = n.pendingProps, s = c.children, r = l !== null ? l.memoizedState : null;
    if (c.mode === "hidden") {
      if ((n.flags & 128) !== 0) {
        if (c = r !== null ? r.baseLanes | u : u, l !== null) {
          for (s = n.child = l.child, r = 0; s !== null; )
            r = r | s.lanes | s.childLanes, s = s.sibling;
          n.childLanes = r & ~c;
        } else n.childLanes = 0, n.child = null;
        return Oc(
          l,
          n,
          c,
          u
        );
      }
      if ((u & 536870912) !== 0)
        n.memoizedState = { baseLanes: 0, cachePool: null }, l !== null && dc(
          n,
          r !== null ? r.cachePool : null
        ), r !== null ? bl(n, r) : Oo(), Kn(n);
      else
        return n.lanes = n.childLanes = 536870912, Oc(
          l,
          n,
          r !== null ? r.baseLanes | u : u,
          u
        );
    } else
      r !== null ? (dc(n, r.cachePool), bl(n, r), Jn(), n.memoizedState = null) : (l !== null && dc(n, null), Oo(), Jn());
    return Sl(l, n, s, u), n.child;
  }
  function Oc(l, n, u, c) {
    var s = Jf();
    return s = s === null ? null : { parent: fl._currentValue, pool: s }, n.memoizedState = {
      baseLanes: u,
      cachePool: s
    }, l !== null && dc(n, null), Oo(), Kn(n), l !== null && To(l, n, c, !0), null;
  }
  function hs(l, n) {
    var u = n.ref;
    if (u === null)
      l !== null && l.ref !== null && (n.flags |= 4194816);
    else {
      if (typeof u != "function" && typeof u != "object")
        throw Error(_(284));
      (l === null || l.ref !== u) && (n.flags |= 4194816);
    }
  }
  function rd(l, n, u, c, s) {
    return mi(n), u = Qr(
      l,
      n,
      u,
      c,
      void 0,
      s
    ), c = Zr(), l !== null && !rl ? (zo(l, n, s), kn(l, n, s)) : (rt && c && sc(n), n.flags |= 1, Sl(l, n, u, s), n.child);
  }
  function Ly(l, n, u, c, s, r) {
    return mi(n), n.updateQueue = null, u = Sy(
      n,
      c,
      u,
      s
    ), bi(l), c = Zr(), l !== null && !rl ? (zo(l, n, r), kn(l, n, r)) : (rt && c && sc(n), n.flags |= 1, Sl(l, n, u, r), n.child);
  }
  function dd(l, n, u, c, s) {
    if (mi(n), n.stateNode === null) {
      var r = mo, y = u.contextType;
      typeof y == "object" && y !== null && (r = gl(y)), r = new u(c, r), n.memoizedState = r.state !== null && r.state !== void 0 ? r.state : null, r.updater = fd, n.stateNode = r, r._reactInternals = n, r = n.stateNode, r.props = c, r.state = n.memoizedState, r.refs = {}, Gr(n), y = u.contextType, r.context = typeof y == "object" && y !== null ? gl(y) : mo, r.state = n.memoizedState, y = u.getDerivedStateFromProps, typeof y == "function" && (Ti(
        n,
        u,
        y,
        c
      ), r.state = n.memoizedState), typeof u.getDerivedStateFromProps == "function" || typeof r.getSnapshotBeforeUpdate == "function" || typeof r.UNSAFE_componentWillMount != "function" && typeof r.componentWillMount != "function" || (y = r.state, typeof r.componentWillMount == "function" && r.componentWillMount(), typeof r.UNSAFE_componentWillMount == "function" && r.UNSAFE_componentWillMount(), y !== r.state && fd.enqueueReplaceState(r, r.state, null), Uu(n, c, r, s), Ro(), r.state = n.memoizedState), typeof r.componentDidMount == "function" && (n.flags |= 4194308), c = !0;
    } else if (l === null) {
      r = n.stateNode;
      var p = n.memoizedProps, S = Ei(u, p);
      r.props = S;
      var H = r.context, K = u.contextType;
      y = mo, typeof K == "object" && K !== null && (y = gl(K));
      var $ = u.getDerivedStateFromProps;
      K = typeof $ == "function" || typeof r.getSnapshotBeforeUpdate == "function", p = n.pendingProps !== p, K || typeof r.UNSAFE_componentWillReceiveProps != "function" && typeof r.componentWillReceiveProps != "function" || (p || H !== y) && Ac(
        n,
        r,
        c,
        y
      ), Ln = !1;
      var q = n.memoizedState;
      r.state = q, Uu(n, c, r, s), Ro(), H = n.memoizedState, p || q !== H || Ln ? (typeof $ == "function" && (Ti(
        n,
        u,
        $,
        c
      ), H = n.memoizedState), (S = Ln || Uo(
        n,
        u,
        S,
        c,
        q,
        H,
        y
      )) ? (K || typeof r.UNSAFE_componentWillMount != "function" && typeof r.componentWillMount != "function" || (typeof r.componentWillMount == "function" && r.componentWillMount(), typeof r.UNSAFE_componentWillMount == "function" && r.UNSAFE_componentWillMount()), typeof r.componentDidMount == "function" && (n.flags |= 4194308)) : (typeof r.componentDidMount == "function" && (n.flags |= 4194308), n.memoizedProps = c, n.memoizedState = H), r.props = c, r.state = H, r.context = y, c = S) : (typeof r.componentDidMount == "function" && (n.flags |= 4194308), c = !1);
    } else {
      r = n.stateNode, Lr(l, n), y = n.memoizedProps, K = Ei(u, y), r.props = K, $ = n.pendingProps, q = r.context, H = u.contextType, S = mo, typeof H == "object" && H !== null && (S = gl(H)), p = u.getDerivedStateFromProps, (H = typeof p == "function" || typeof r.getSnapshotBeforeUpdate == "function") || typeof r.UNSAFE_componentWillReceiveProps != "function" && typeof r.componentWillReceiveProps != "function" || (y !== $ || q !== S) && Ac(
        n,
        r,
        c,
        S
      ), Ln = !1, q = n.memoizedState, r.state = q, Uu(n, c, r, s), Ro();
      var Y = n.memoizedState;
      y !== $ || q !== Y || Ln || l !== null && l.dependencies !== null && Qf(l.dependencies) ? (typeof p == "function" && (Ti(
        n,
        u,
        p,
        c
      ), Y = n.memoizedState), (K = Ln || Uo(
        n,
        u,
        K,
        c,
        q,
        Y,
        S
      ) || l !== null && l.dependencies !== null && Qf(l.dependencies)) ? (H || typeof r.UNSAFE_componentWillUpdate != "function" && typeof r.componentWillUpdate != "function" || (typeof r.componentWillUpdate == "function" && r.componentWillUpdate(c, Y, S), typeof r.UNSAFE_componentWillUpdate == "function" && r.UNSAFE_componentWillUpdate(
        c,
        Y,
        S
      )), typeof r.componentDidUpdate == "function" && (n.flags |= 4), typeof r.getSnapshotBeforeUpdate == "function" && (n.flags |= 1024)) : (typeof r.componentDidUpdate != "function" || y === l.memoizedProps && q === l.memoizedState || (n.flags |= 4), typeof r.getSnapshotBeforeUpdate != "function" || y === l.memoizedProps && q === l.memoizedState || (n.flags |= 1024), n.memoizedProps = c, n.memoizedState = Y), r.props = c, r.state = Y, r.context = S, c = K) : (typeof r.componentDidUpdate != "function" || y === l.memoizedProps && q === l.memoizedState || (n.flags |= 4), typeof r.getSnapshotBeforeUpdate != "function" || y === l.memoizedProps && q === l.memoizedState || (n.flags |= 1024), c = !1);
    }
    return r = c, hs(l, n), c = (n.flags & 128) !== 0, r || c ? (r = n.stateNode, u = c && typeof u.getDerivedStateFromError != "function" ? null : r.render(), n.flags |= 1, l !== null && c ? (n.child = Ec(
      n,
      l.child,
      null,
      s
    ), n.child = Ec(
      n,
      null,
      u,
      s
    )) : Sl(l, n, u, s), n.memoizedState = r.state, l = n.child) : l = kn(
      l,
      n,
      s
    ), l;
  }
  function hd(l, n, u, c) {
    return bo(), n.flags |= 256, Sl(l, n, u, c), n.child;
  }
  var yd = {
    dehydrated: null,
    treeContext: null,
    retryLane: 0,
    hydrationErrors: null
  };
  function Vy(l) {
    return { baseLanes: l, cachePool: qr() };
  }
  function Xy(l, n, u) {
    return l = l !== null ? l.childLanes & ~u : 0, n && (l |= Fa), l;
  }
  function Qy(l, n, u) {
    var c = n.pendingProps, s = !1, r = (n.flags & 128) !== 0, y;
    if ((y = r) || (y = l !== null && l.memoizedState === null ? !1 : (zt.current & 2) !== 0), y && (s = !0, n.flags &= -129), y = (n.flags & 32) !== 0, n.flags &= -33, l === null) {
      if (rt) {
        if (s ? Hu(n) : Jn(), rt) {
          var p = dt, S;
          if (S = p) {
            e: {
              for (S = p, p = Za; S.nodeType !== 8; ) {
                if (!p) {
                  p = null;
                  break e;
                }
                if (S = Tn(
                  S.nextSibling
                ), S === null) {
                  p = null;
                  break e;
                }
              }
              p = S;
            }
            p !== null ? (n.memoizedState = {
              dehydrated: p,
              treeContext: Du !== null ? { id: mn, overflow: Qt } : null,
              retryLane: 536870912,
              hydrationErrors: null
            }, S = oa(
              18,
              null,
              null,
              0
            ), S.stateNode = p, S.return = n, n.child = S, Pt = n, dt = null, S = !0) : S = !1;
          }
          S || zu(n);
        }
        if (p = n.memoizedState, p !== null && (p = p.dehydrated, p !== null))
          return xs(p) ? n.lanes = 32 : n.lanes = 536870912, null;
        bn(n);
      }
      return p = c.children, c = c.fallback, s ? (Jn(), s = n.mode, p = pd(
        { mode: "hidden", children: p },
        s
      ), c = Va(
        c,
        s,
        u,
        null
      ), p.return = n, c.return = n, p.sibling = c, n.child = p, s = n.child, s.memoizedState = Vy(u), s.childLanes = Xy(
        l,
        y,
        u
      ), n.memoizedState = yd, c) : (Hu(n), md(n, p));
    }
    if (S = l.memoizedState, S !== null && (p = S.dehydrated, p !== null)) {
      if (r)
        n.flags & 256 ? (Hu(n), n.flags &= -257, n = Ai(
          l,
          n,
          u
        )) : n.memoizedState !== null ? (Jn(), n.child = l.child, n.flags |= 128, n = null) : (Jn(), s = c.fallback, p = n.mode, c = pd(
          { mode: "visible", children: c.children },
          p
        ), s = Va(
          s,
          p,
          u,
          null
        ), s.flags |= 2, c.return = n, s.return = n, c.sibling = s, n.child = c, Ec(
          n,
          l.child,
          null,
          u
        ), c = n.child, c.memoizedState = Vy(u), c.childLanes = Xy(
          l,
          y,
          u
        ), n.memoizedState = yd, n = s);
      else if (Hu(n), xs(p)) {
        if (y = p.nextSibling && p.nextSibling.dataset, y) var H = y.dgst;
        y = H, c = Error(_(419)), c.stack = "", c.digest = y, So({ value: c, source: null, stack: null }), n = Ai(
          l,
          n,
          u
        );
      } else if (rl || To(l, n, u, !1), y = (u & l.childLanes) !== 0, rl || y) {
        if (y = _t, y !== null && (c = u & -u, c = (c & 42) !== 0 ? 1 : ll(c), c = (c & (y.suspendedLanes | u)) !== 0 ? 0 : c, c !== 0 && c !== S.retryLane))
          throw S.retryLane = c, qn(l, c), Ua(y, l, c), Kt;
        p.data === "$?" || xc(), n = Ai(
          l,
          n,
          u
        );
      } else
        p.data === "$?" ? (n.flags |= 192, n.child = l.child, n = null) : (l = S.treeContext, dt = Tn(
          p.nextSibling
        ), Pt = n, rt = !0, Qa = null, Za = !1, l !== null && (Xa[fa++] = mn, Xa[fa++] = Qt, Xa[fa++] = Du, mn = l.id, Qt = l.overflow, Du = n), n = md(
          n,
          c.children
        ), n.flags |= 4096);
      return n;
    }
    return s ? (Jn(), s = c.fallback, p = n.mode, S = l.child, H = S.sibling, c = yn(S, {
      mode: "hidden",
      children: c.children
    }), c.subtreeFlags = S.subtreeFlags & 65011712, H !== null ? s = yn(H, s) : (s = Va(
      s,
      p,
      u,
      null
    ), s.flags |= 2), s.return = n, c.return = n, c.sibling = s, n.child = c, c = s, s = n.child, p = l.child.memoizedState, p === null ? p = Vy(u) : (S = p.cachePool, S !== null ? (H = fl._currentValue, S = S.parent !== H ? { parent: H, pool: H } : S) : S = qr(), p = {
      baseLanes: p.baseLanes | u,
      cachePool: S
    }), s.memoizedState = p, s.childLanes = Xy(
      l,
      y,
      u
    ), n.memoizedState = yd, c) : (Hu(n), u = l.child, l = u.sibling, u = yn(u, {
      mode: "visible",
      children: c.children
    }), u.return = n, u.sibling = null, l !== null && (y = n.deletions, y === null ? (n.deletions = [l], n.flags |= 16) : y.push(l)), n.child = u, n.memoizedState = null, u);
  }
  function md(l, n) {
    return n = pd(
      { mode: "visible", children: n },
      l.mode
    ), n.return = l, l.child = n;
  }
  function pd(l, n) {
    return l = oa(22, l, null, n), l.lanes = 0, l.stateNode = {
      _visibility: 1,
      _pendingMarkers: null,
      _retryCache: null,
      _transitions: null
    }, l;
  }
  function Ai(l, n, u) {
    return Ec(n, l.child, null, u), l = md(
      n,
      n.pendingProps.children
    ), l.flags |= 2, n.memoizedState = null, l;
  }
  function ys(l, n, u) {
    l.lanes |= n;
    var c = l.alternate;
    c !== null && (c.lanes |= n), Cr(l.return, n, u);
  }
  function vd(l, n, u, c, s) {
    var r = l.memoizedState;
    r === null ? l.memoizedState = {
      isBackwards: n,
      rendering: null,
      renderingStartTime: 0,
      last: c,
      tail: u,
      tailMode: s
    } : (r.isBackwards = n, r.rendering = null, r.renderingStartTime = 0, r.last = c, r.tail = u, r.tailMode = s);
  }
  function gd(l, n, u) {
    var c = n.pendingProps, s = c.revealOrder, r = c.tail;
    if (Sl(l, n, c.children, u), c = zt.current, (c & 2) !== 0)
      c = c & 1 | 2, n.flags |= 128;
    else {
      if (l !== null && (l.flags & 128) !== 0)
        e: for (l = n.child; l !== null; ) {
          if (l.tag === 13)
            l.memoizedState !== null && ys(l, u, n);
          else if (l.tag === 19)
            ys(l, u, n);
          else if (l.child !== null) {
            l.child.return = l, l = l.child;
            continue;
          }
          if (l === n) break e;
          for (; l.sibling === null; ) {
            if (l.return === null || l.return === n)
              break e;
            l = l.return;
          }
          l.sibling.return = l.return, l = l.sibling;
        }
      c &= 1;
    }
    switch (I(zt, c), s) {
      case "forwards":
        for (u = n.child, s = null; u !== null; )
          l = u.alternate, l !== null && fs(l) === null && (s = u), u = u.sibling;
        u = s, u === null ? (s = n.child, n.child = null) : (s = u.sibling, u.sibling = null), vd(
          n,
          !1,
          s,
          u,
          r
        );
        break;
      case "backwards":
        for (u = null, s = n.child, n.child = null; s !== null; ) {
          if (l = s.alternate, l !== null && fs(l) === null) {
            n.child = s;
            break;
          }
          l = s.sibling, s.sibling = u, u = s, s = l;
        }
        vd(
          n,
          !0,
          u,
          null,
          r
        );
        break;
      case "together":
        vd(n, !1, null, null, void 0);
        break;
      default:
        n.memoizedState = null;
    }
    return n.child;
  }
  function kn(l, n, u) {
    if (l !== null && (n.dependencies = l.dependencies), ju |= n.lanes, (u & n.childLanes) === 0)
      if (l !== null) {
        if (To(
          l,
          n,
          u,
          !1
        ), (u & n.childLanes) === 0)
          return null;
      } else return null;
    if (l !== null && n.child !== l.child)
      throw Error(_(153));
    if (n.child !== null) {
      for (l = n.child, u = yn(l, l.pendingProps), n.child = u, u.return = n; l.sibling !== null; )
        l = l.sibling, u = u.sibling = yn(l, l.pendingProps), u.return = n;
      u.sibling = null;
    }
    return n.child;
  }
  function bd(l, n) {
    return (l.lanes & n) !== 0 ? !0 : (l = l.dependencies, !!(l !== null && Qf(l)));
  }
  function cv(l, n, u) {
    switch (n.tag) {
      case 3:
        He(n, n.stateNode.containerInfo), _u(n, fl, l.memoizedState.cache), bo();
        break;
      case 27:
      case 5:
        na(n);
        break;
      case 4:
        He(n, n.stateNode.containerInfo);
        break;
      case 10:
        _u(
          n,
          n.type,
          n.memoizedProps.value
        );
        break;
      case 13:
        var c = n.memoizedState;
        if (c !== null)
          return c.dehydrated !== null ? (Hu(n), n.flags |= 128, null) : (u & n.child.childLanes) !== 0 ? Qy(l, n, u) : (Hu(n), l = kn(
            l,
            n,
            u
          ), l !== null ? l.sibling : null);
        Hu(n);
        break;
      case 19:
        var s = (l.flags & 128) !== 0;
        if (c = (u & n.childLanes) !== 0, c || (To(
          l,
          n,
          u,
          !1
        ), c = (u & n.childLanes) !== 0), s) {
          if (c)
            return gd(
              l,
              n,
              u
            );
          n.flags |= 128;
        }
        if (s = n.memoizedState, s !== null && (s.rendering = null, s.tail = null, s.lastEffect = null), I(zt, zt.current), c) break;
        return null;
      case 22:
      case 23:
        return n.lanes = 0, sd(l, n, u);
      case 24:
        _u(n, fl, l.memoizedState.cache);
    }
    return kn(l, n, u);
  }
  function ov(l, n, u) {
    if (l !== null)
      if (l.memoizedProps !== n.pendingProps)
        rl = !0;
      else {
        if (!bd(l, u) && (n.flags & 128) === 0)
          return rl = !1, cv(
            l,
            n,
            u
          );
        rl = (l.flags & 131072) !== 0;
      }
    else
      rl = !1, rt && (n.flags & 1048576) !== 0 && Ur(n, vo, n.index);
    switch (n.lanes = 0, n.tag) {
      case 16:
        e: {
          l = n.pendingProps;
          var c = n.elementType, s = c._init;
          if (c = s(c._payload), n.type = c, typeof c == "function")
            Gf(c) ? (l = Ei(c, l), n.tag = 1, n = dd(
              null,
              n,
              c,
              l,
              u
            )) : (n.tag = 0, n = rd(
              null,
              n,
              c,
              l,
              u
            ));
          else {
            if (c != null) {
              if (s = c.$$typeof, s === Nt) {
                n.tag = 11, n = iv(
                  null,
                  n,
                  c,
                  l,
                  u
                );
                break e;
              } else if (s === De) {
                n.tag = 14, n = Nu(
                  null,
                  n,
                  c,
                  l,
                  u
                );
                break e;
              }
            }
            throw n = Gt(c) || c, Error(_(306, n, ""));
          }
        }
        return n;
      case 0:
        return rd(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 1:
        return c = n.type, s = Ei(
          c,
          n.pendingProps
        ), dd(
          l,
          n,
          c,
          s,
          u
        );
      case 3:
        e: {
          if (He(
            n,
            n.stateNode.containerInfo
          ), l === null) throw Error(_(387));
          c = n.pendingProps;
          var r = n.memoizedState;
          s = r.element, Lr(l, n), Uu(n, c, null, u);
          var y = n.memoizedState;
          if (c = y.cache, _u(n, fl, c), c !== r.cache && dy(
            n,
            [fl],
            u,
            !0
          ), Ro(), c = y.element, r.isDehydrated)
            if (r = {
              element: c,
              isDehydrated: !1,
              cache: y.cache
            }, n.updateQueue.baseState = r, n.memoizedState = r, n.flags & 256) {
              n = hd(
                l,
                n,
                c,
                u
              );
              break e;
            } else if (c !== s) {
              s = Oa(
                Error(_(424)),
                n
              ), So(s), n = hd(
                l,
                n,
                c,
                u
              );
              break e;
            } else {
              switch (l = n.stateNode.containerInfo, l.nodeType) {
                case 9:
                  l = l.body;
                  break;
                default:
                  l = l.nodeName === "HTML" ? l.ownerDocument.body : l;
              }
              for (dt = Tn(l.firstChild), Pt = n, rt = !0, Qa = null, Za = !0, u = Zn(
                n,
                null,
                c,
                u
              ), n.child = u; u; )
                u.flags = u.flags & -3 | 4096, u = u.sibling;
            }
          else {
            if (bo(), c === s) {
              n = kn(
                l,
                n,
                u
              );
              break e;
            }
            Sl(
              l,
              n,
              c,
              u
            );
          }
          n = n.child;
        }
        return n;
      case 26:
        return hs(l, n), l === null ? (u = Rv(
          n.type,
          null,
          n.pendingProps,
          null
        )) ? n.memoizedState = u : rt || (u = n.type, l = n.pendingProps, c = Pa(
          oe.current
        ).createElement(u), c[vl] = n, c[kl] = l, Ce(c, u, l), ol(c), n.stateNode = c) : n.memoizedState = Rv(
          n.type,
          l.memoizedProps,
          n.pendingProps,
          l.memoizedState
        ), null;
      case 27:
        return na(n), l === null && rt && (c = n.stateNode = fe(
          n.type,
          n.pendingProps,
          oe.current
        ), Pt = n, Za = !0, s = dt, xi(n.type) ? (Hi = s, dt = Tn(
          c.firstChild
        )) : dt = s), Sl(
          l,
          n,
          n.pendingProps.children,
          u
        ), hs(l, n), l === null && (n.flags |= 4194304), n.child;
      case 5:
        return l === null && rt && ((s = c = dt) && (c = Wo(
          c,
          n.type,
          n.pendingProps,
          Za
        ), c !== null ? (n.stateNode = c, Pt = n, dt = Tn(
          c.firstChild
        ), Za = !1, s = !0) : s = !1), s || zu(n)), na(n), s = n.type, r = n.pendingProps, y = l !== null ? l.memoizedProps : null, c = r.children, au(s, r) ? c = null : y !== null && au(s, y) && (n.flags |= 32), n.memoizedState !== null && (s = Qr(
          l,
          n,
          Wp,
          null,
          null,
          u
        ), ba._currentValue = s), hs(l, n), Sl(l, n, c, u), n.child;
      case 6:
        return l === null && rt && ((l = u = dt) && (u = Bg(
          u,
          n.pendingProps,
          Za
        ), u !== null ? (n.stateNode = u, Pt = n, dt = null, l = !0) : l = !1), l || zu(n)), null;
      case 13:
        return Qy(l, n, u);
      case 4:
        return He(
          n,
          n.stateNode.containerInfo
        ), c = n.pendingProps, l === null ? n.child = Ec(
          n,
          null,
          c,
          u
        ) : Sl(
          l,
          n,
          c,
          u
        ), n.child;
      case 11:
        return iv(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 7:
        return Sl(
          l,
          n,
          n.pendingProps,
          u
        ), n.child;
      case 8:
        return Sl(
          l,
          n,
          n.pendingProps.children,
          u
        ), n.child;
      case 12:
        return Sl(
          l,
          n,
          n.pendingProps.children,
          u
        ), n.child;
      case 10:
        return c = n.pendingProps, _u(n, n.type, c.value), Sl(
          l,
          n,
          c.children,
          u
        ), n.child;
      case 9:
        return s = n.type._context, c = n.pendingProps.children, mi(n), s = gl(s), c = c(s), n.flags |= 1, Sl(l, n, c, u), n.child;
      case 14:
        return Nu(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 15:
        return Rc(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 19:
        return gd(l, n, u);
      case 31:
        return c = n.pendingProps, u = n.mode, c = {
          mode: c.mode,
          children: c.children
        }, l === null ? (u = pd(
          c,
          u
        ), u.ref = n.ref, n.child = u, u.return = n, n = u) : (u = yn(l.child, c), u.ref = n.ref, n.child = u, u.return = n, n = u), n;
      case 22:
        return sd(l, n, u);
      case 24:
        return mi(n), c = gl(fl), l === null ? (s = Jf(), s === null && (s = _t, r = Ao(), s.pooledCache = r, r.refCount++, r !== null && (s.pooledCacheLanes |= u), s = r), n.memoizedState = {
          parent: c,
          cache: s
        }, Gr(n), _u(n, fl, s)) : ((l.lanes & u) !== 0 && (Lr(l, n), Uu(n, null, null, u), Ro()), s = l.memoizedState, r = n.memoizedState, s.parent !== c ? (s = { parent: c, cache: c }, n.memoizedState = s, n.lanes === 0 && (n.memoizedState = n.updateQueue.baseState = s), _u(n, fl, c)) : (c = r.cache, _u(n, fl, c), c !== s.cache && dy(
          n,
          [fl],
          u,
          !0
        ))), Sl(
          l,
          n,
          n.pendingProps.children,
          u
        ), n.child;
      case 29:
        throw n.pendingProps;
    }
    throw Error(_(156, n.tag));
  }
  function $n(l) {
    l.flags |= 4;
  }
  function xo(l, n) {
    if (n.type !== "stylesheet" || (n.state.loading & 4) !== 0)
      l.flags &= -16777217;
    else if (l.flags |= 16777216, !zm(n)) {
      if (n = Ma.current, n !== null && ((lt & 4194048) === lt ? Xl !== null : (lt & 62914560) !== lt && (lt & 536870912) === 0 || n !== Xl))
        throw hc = Br, jr;
      l.flags |= 8192;
    }
  }
  function ms(l, n) {
    n !== null && (l.flags |= 4), l.flags & 16384 && (n = l.tag !== 22 ? ue() : 536870912, l.lanes |= n, jo |= n);
  }
  function Ho(l, n) {
    if (!rt)
      switch (l.tailMode) {
        case "hidden":
          n = l.tail;
          for (var u = null; n !== null; )
            n.alternate !== null && (u = n), n = n.sibling;
          u === null ? l.tail = null : u.sibling = null;
          break;
        case "collapsed":
          u = l.tail;
          for (var c = null; u !== null; )
            u.alternate !== null && (c = u), u = u.sibling;
          c === null ? n || l.tail === null ? l.tail = null : l.tail.sibling = null : c.sibling = null;
      }
  }
  function Me(l) {
    var n = l.alternate !== null && l.alternate.child === l.child, u = 0, c = 0;
    if (n)
      for (var s = l.child; s !== null; )
        u |= s.lanes | s.childLanes, c |= s.subtreeFlags & 65011712, c |= s.flags & 65011712, s.return = l, s = s.sibling;
    else
      for (s = l.child; s !== null; )
        u |= s.lanes | s.childLanes, c |= s.subtreeFlags, c |= s.flags, s.return = l, s = s.sibling;
    return l.subtreeFlags |= c, l.childLanes = u, n;
  }
  function Zy(l, n, u) {
    var c = n.pendingProps;
    switch (jn(n), n.tag) {
      case 31:
      case 16:
      case 15:
      case 0:
      case 11:
      case 7:
      case 8:
      case 12:
      case 9:
      case 14:
        return Me(n), null;
      case 1:
        return Me(n), null;
      case 3:
        return u = n.stateNode, c = null, l !== null && (c = l.memoizedState.cache), n.memoizedState.cache !== c && (n.flags |= 2048), Bn(fl), wt(), u.pendingContext && (u.context = u.pendingContext, u.pendingContext = null), (l === null || l.child === null) && (go(n) ? $n(n) : l === null || l.memoizedState.isDehydrated && (n.flags & 256) === 0 || (n.flags |= 1024, ry())), Me(n), null;
      case 26:
        return u = n.memoizedState, l === null ? ($n(n), u !== null ? (Me(n), xo(n, u)) : (Me(n), n.flags &= -16777217)) : u ? u !== l.memoizedState ? ($n(n), Me(n), xo(n, u)) : (Me(n), n.flags &= -16777217) : (l.memoizedProps !== c && $n(n), Me(n), n.flags &= -16777217), null;
      case 27:
        Dn(n), u = oe.current;
        var s = n.type;
        if (l !== null && n.stateNode != null)
          l.memoizedProps !== c && $n(n);
        else {
          if (!c) {
            if (n.stateNode === null)
              throw Error(_(166));
            return Me(n), null;
          }
          l = ce.current, go(n) ? Vf(n) : (l = fe(s, c, u), n.stateNode = l, $n(n));
        }
        return Me(n), null;
      case 5:
        if (Dn(n), u = n.type, l !== null && n.stateNode != null)
          l.memoizedProps !== c && $n(n);
        else {
          if (!c) {
            if (n.stateNode === null)
              throw Error(_(166));
            return Me(n), null;
          }
          if (l = ce.current, go(n))
            Vf(n);
          else {
            switch (s = Pa(
              oe.current
            ), l) {
              case 1:
                l = s.createElementNS(
                  "http://www.w3.org/2000/svg",
                  u
                );
                break;
              case 2:
                l = s.createElementNS(
                  "http://www.w3.org/1998/Math/MathML",
                  u
                );
                break;
              default:
                switch (u) {
                  case "svg":
                    l = s.createElementNS(
                      "http://www.w3.org/2000/svg",
                      u
                    );
                    break;
                  case "math":
                    l = s.createElementNS(
                      "http://www.w3.org/1998/Math/MathML",
                      u
                    );
                    break;
                  case "script":
                    l = s.createElement("div"), l.innerHTML = "<script><\/script>", l = l.removeChild(l.firstChild);
                    break;
                  case "select":
                    l = typeof c.is == "string" ? s.createElement("select", { is: c.is }) : s.createElement("select"), c.multiple ? l.multiple = !0 : c.size && (l.size = c.size);
                    break;
                  default:
                    l = typeof c.is == "string" ? s.createElement(u, { is: c.is }) : s.createElement(u);
                }
            }
            l[vl] = n, l[kl] = c;
            e: for (s = n.child; s !== null; ) {
              if (s.tag === 5 || s.tag === 6)
                l.appendChild(s.stateNode);
              else if (s.tag !== 4 && s.tag !== 27 && s.child !== null) {
                s.child.return = s, s = s.child;
                continue;
              }
              if (s === n) break e;
              for (; s.sibling === null; ) {
                if (s.return === null || s.return === n)
                  break e;
                s = s.return;
              }
              s.sibling.return = s.return, s = s.sibling;
            }
            n.stateNode = l;
            e: switch (Ce(l, u, c), u) {
              case "button":
              case "input":
              case "select":
              case "textarea":
                l = !!c.autoFocus;
                break e;
              case "img":
                l = !0;
                break e;
              default:
                l = !1;
            }
            l && $n(n);
          }
        }
        return Me(n), n.flags &= -16777217, null;
      case 6:
        if (l && n.stateNode != null)
          l.memoizedProps !== c && $n(n);
        else {
          if (typeof c != "string" && n.stateNode === null)
            throw Error(_(166));
          if (l = oe.current, go(n)) {
            if (l = n.stateNode, u = n.memoizedProps, c = null, s = Pt, s !== null)
              switch (s.tag) {
                case 27:
                case 5:
                  c = s.memoizedProps;
              }
            l[vl] = n, l = !!(l.nodeValue === u || c !== null && c.suppressHydrationWarning === !0 || Am(l.nodeValue, u)), l || zu(n);
          } else
            l = Pa(l).createTextNode(
              c
            ), l[vl] = n, n.stateNode = l;
        }
        return Me(n), null;
      case 13:
        if (c = n.memoizedState, l === null || l.memoizedState !== null && l.memoizedState.dehydrated !== null) {
          if (s = go(n), c !== null && c.dehydrated !== null) {
            if (l === null) {
              if (!s) throw Error(_(318));
              if (s = n.memoizedState, s = s !== null ? s.dehydrated : null, !s) throw Error(_(317));
              s[vl] = n;
            } else
              bo(), (n.flags & 128) === 0 && (n.memoizedState = null), n.flags |= 4;
            Me(n), s = !1;
          } else
            s = ry(), l !== null && l.memoizedState !== null && (l.memoizedState.hydrationErrors = s), s = !0;
          if (!s)
            return n.flags & 256 ? (bn(n), n) : (bn(n), null);
        }
        if (bn(n), (n.flags & 128) !== 0)
          return n.lanes = u, n;
        if (u = c !== null, l = l !== null && l.memoizedState !== null, u) {
          c = n.child, s = null, c.alternate !== null && c.alternate.memoizedState !== null && c.alternate.memoizedState.cachePool !== null && (s = c.alternate.memoizedState.cachePool.pool);
          var r = null;
          c.memoizedState !== null && c.memoizedState.cachePool !== null && (r = c.memoizedState.cachePool.pool), r !== s && (c.flags |= 2048);
        }
        return u !== l && u && (n.child.flags |= 8192), ms(n, n.updateQueue), Me(n), null;
      case 4:
        return wt(), l === null && Tm(n.stateNode.containerInfo), Me(n), null;
      case 10:
        return Bn(n.type), Me(n), null;
      case 19:
        if (J(zt), s = n.memoizedState, s === null) return Me(n), null;
        if (c = (n.flags & 128) !== 0, r = s.rendering, r === null)
          if (c) Ho(s, !1);
          else {
            if ($t !== 0 || l !== null && (l.flags & 128) !== 0)
              for (l = n.child; l !== null; ) {
                if (r = fs(l), r !== null) {
                  for (n.flags |= 128, Ho(s, !1), l = r.updateQueue, n.updateQueue = l, ms(n, l), n.subtreeFlags = 0, l = u, u = n.child; u !== null; )
                    $e(u, l), u = u.sibling;
                  return I(
                    zt,
                    zt.current & 1 | 2
                  ), n.child;
                }
                l = l.sibling;
              }
            s.tail !== null && pl() > Md && (n.flags |= 128, c = !0, Ho(s, !1), n.lanes = 4194304);
          }
        else {
          if (!c)
            if (l = fs(r), l !== null) {
              if (n.flags |= 128, c = !0, l = l.updateQueue, n.updateQueue = l, ms(n, l), Ho(s, !0), s.tail === null && s.tailMode === "hidden" && !r.alternate && !rt)
                return Me(n), null;
            } else
              2 * pl() - s.renderingStartTime > Md && u !== 536870912 && (n.flags |= 128, c = !0, Ho(s, !1), n.lanes = 4194304);
          s.isBackwards ? (r.sibling = n.child, n.child = r) : (l = s.last, l !== null ? l.sibling = r : n.child = r, s.last = r);
        }
        return s.tail !== null ? (n = s.tail, s.rendering = n, s.tail = n.sibling, s.renderingStartTime = pl(), n.sibling = null, l = zt.current, I(zt, c ? l & 1 | 2 : l & 1), n) : (Me(n), null);
      case 22:
      case 23:
        return bn(n), Do(), c = n.memoizedState !== null, l !== null ? l.memoizedState !== null !== c && (n.flags |= 8192) : c && (n.flags |= 8192), c ? (u & 536870912) !== 0 && (n.flags & 128) === 0 && (Me(n), n.subtreeFlags & 6 && (n.flags |= 8192)) : Me(n), u = n.updateQueue, u !== null && ms(n, u.retryQueue), u = null, l !== null && l.memoizedState !== null && l.memoizedState.cachePool !== null && (u = l.memoizedState.cachePool.pool), c = null, n.memoizedState !== null && n.memoizedState.cachePool !== null && (c = n.memoizedState.cachePool.pool), c !== u && (n.flags |= 2048), l !== null && J(Gn), null;
      case 24:
        return u = null, l !== null && (u = l.memoizedState.cache), n.memoizedState.cache !== u && (n.flags |= 2048), Bn(fl), Me(n), null;
      case 25:
        return null;
      case 30:
        return null;
    }
    throw Error(_(156, n.tag));
  }
  function Ug(l, n) {
    switch (jn(n), n.tag) {
      case 1:
        return l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 3:
        return Bn(fl), wt(), l = n.flags, (l & 65536) !== 0 && (l & 128) === 0 ? (n.flags = l & -65537 | 128, n) : null;
      case 26:
      case 27:
      case 5:
        return Dn(n), null;
      case 13:
        if (bn(n), l = n.memoizedState, l !== null && l.dehydrated !== null) {
          if (n.alternate === null)
            throw Error(_(340));
          bo();
        }
        return l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 19:
        return J(zt), null;
      case 4:
        return wt(), null;
      case 10:
        return Bn(n.type), null;
      case 22:
      case 23:
        return bn(n), Do(), l !== null && J(Gn), l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 24:
        return Bn(fl), null;
      case 25:
        return null;
      default:
        return null;
    }
  }
  function Ky(l, n) {
    switch (jn(n), n.tag) {
      case 3:
        Bn(fl), wt();
        break;
      case 26:
      case 27:
      case 5:
        Dn(n);
        break;
      case 4:
        wt();
        break;
      case 13:
        bn(n);
        break;
      case 19:
        J(zt);
        break;
      case 10:
        Bn(n.type);
        break;
      case 22:
      case 23:
        bn(n), Do(), l !== null && J(Gn);
        break;
      case 24:
        Bn(fl);
    }
  }
  function ps(l, n) {
    try {
      var u = n.updateQueue, c = u !== null ? u.lastEffect : null;
      if (c !== null) {
        var s = c.next;
        u = s;
        do {
          if ((u.tag & l) === l) {
            c = void 0;
            var r = u.create, y = u.inst;
            c = r(), y.destroy = c;
          }
          u = u.next;
        } while (u !== s);
      }
    } catch (p) {
      Tt(n, n.return, p);
    }
  }
  function Ri(l, n, u) {
    try {
      var c = n.updateQueue, s = c !== null ? c.lastEffect : null;
      if (s !== null) {
        var r = s.next;
        c = r;
        do {
          if ((c.tag & l) === l) {
            var y = c.inst, p = y.destroy;
            if (p !== void 0) {
              y.destroy = void 0, s = n;
              var S = u, H = p;
              try {
                H();
              } catch (K) {
                Tt(
                  s,
                  S,
                  K
                );
              }
            }
          }
          c = c.next;
        } while (c !== r);
      }
    } catch (K) {
      Tt(n, n.return, K);
    }
  }
  function Sd(l) {
    var n = l.updateQueue;
    if (n !== null) {
      var u = l.stateNode;
      try {
        Wf(n, u);
      } catch (c) {
        Tt(l, l.return, c);
      }
    }
  }
  function Jy(l, n, u) {
    u.props = Ei(
      l.type,
      l.memoizedProps
    ), u.state = l.memoizedState;
    try {
      u.componentWillUnmount();
    } catch (c) {
      Tt(l, n, c);
    }
  }
  function No(l, n) {
    try {
      var u = l.ref;
      if (u !== null) {
        switch (l.tag) {
          case 26:
          case 27:
          case 5:
            var c = l.stateNode;
            break;
          case 30:
            c = l.stateNode;
            break;
          default:
            c = l.stateNode;
        }
        typeof u == "function" ? l.refCleanup = u(c) : u.current = c;
      }
    } catch (s) {
      Tt(l, n, s);
    }
  }
  function Sn(l, n) {
    var u = l.ref, c = l.refCleanup;
    if (u !== null)
      if (typeof c == "function")
        try {
          c();
        } catch (s) {
          Tt(l, n, s);
        } finally {
          l.refCleanup = null, l = l.alternate, l != null && (l.refCleanup = null);
        }
      else if (typeof u == "function")
        try {
          u(null);
        } catch (s) {
          Tt(l, n, s);
        }
      else u.current = null;
  }
  function wo(l) {
    var n = l.type, u = l.memoizedProps, c = l.stateNode;
    try {
      e: switch (n) {
        case "button":
        case "input":
        case "select":
        case "textarea":
          u.autoFocus && c.focus();
          break e;
        case "img":
          u.src ? c.src = u.src : u.srcSet && (c.srcset = u.srcSet);
      }
    } catch (s) {
      Tt(l, l.return, s);
    }
  }
  function ky(l, n, u) {
    try {
      var c = l.stateNode;
      wg(c, l.type, u, n), c[kl] = n;
    } catch (s) {
      Tt(l, l.return, s);
    }
  }
  function fv(l) {
    return l.tag === 5 || l.tag === 3 || l.tag === 26 || l.tag === 27 && xi(l.type) || l.tag === 4;
  }
  function $a(l) {
    e: for (; ; ) {
      for (; l.sibling === null; ) {
        if (l.return === null || fv(l.return)) return null;
        l = l.return;
      }
      for (l.sibling.return = l.return, l = l.sibling; l.tag !== 5 && l.tag !== 6 && l.tag !== 18; ) {
        if (l.tag === 27 && xi(l.type) || l.flags & 2 || l.child === null || l.tag === 4) continue e;
        l.child.return = l, l = l.child;
      }
      if (!(l.flags & 2)) return l.stateNode;
    }
  }
  function Dc(l, n, u) {
    var c = l.tag;
    if (c === 5 || c === 6)
      l = l.stateNode, n ? (u.nodeType === 9 ? u.body : u.nodeName === "HTML" ? u.ownerDocument.body : u).insertBefore(l, n) : (n = u.nodeType === 9 ? u.body : u.nodeName === "HTML" ? u.ownerDocument.body : u, n.appendChild(l), u = u._reactRootContainer, u != null || n.onclick !== null || (n.onclick = Gd));
    else if (c !== 4 && (c === 27 && xi(l.type) && (u = l.stateNode, n = null), l = l.child, l !== null))
      for (Dc(l, n, u), l = l.sibling; l !== null; )
        Dc(l, n, u), l = l.sibling;
  }
  function Td(l, n, u) {
    var c = l.tag;
    if (c === 5 || c === 6)
      l = l.stateNode, n ? u.insertBefore(l, n) : u.appendChild(l);
    else if (c !== 4 && (c === 27 && xi(l.type) && (u = l.stateNode), l = l.child, l !== null))
      for (Td(l, n, u), l = l.sibling; l !== null; )
        Td(l, n, u), l = l.sibling;
  }
  function Ed(l) {
    var n = l.stateNode, u = l.memoizedProps;
    try {
      for (var c = l.type, s = n.attributes; s.length; )
        n.removeAttributeNode(s[0]);
      Ce(n, c, u), n[vl] = l, n[kl] = u;
    } catch (r) {
      Tt(l, l.return, r);
    }
  }
  var Wn = !1, Jt = !1, Ad = !1, Rd = typeof WeakSet == "function" ? WeakSet : Set, dl = null;
  function $y(l, n) {
    if (l = l.containerInfo, _s = ws, l = ny(l), Bf(l)) {
      if ("selectionStart" in l)
        var u = {
          start: l.selectionStart,
          end: l.selectionEnd
        };
      else
        e: {
          u = (u = l.ownerDocument) && u.defaultView || window;
          var c = u.getSelection && u.getSelection();
          if (c && c.rangeCount !== 0) {
            u = c.anchorNode;
            var s = c.anchorOffset, r = c.focusNode;
            c = c.focusOffset;
            try {
              u.nodeType, r.nodeType;
            } catch {
              u = null;
              break e;
            }
            var y = 0, p = -1, S = -1, H = 0, K = 0, $ = l, q = null;
            t: for (; ; ) {
              for (var Y; $ !== u || s !== 0 && $.nodeType !== 3 || (p = y + s), $ !== r || c !== 0 && $.nodeType !== 3 || (S = y + c), $.nodeType === 3 && (y += $.nodeValue.length), (Y = $.firstChild) !== null; )
                q = $, $ = Y;
              for (; ; ) {
                if ($ === l) break t;
                if (q === u && ++H === s && (p = y), q === r && ++K === c && (S = y), (Y = $.nextSibling) !== null) break;
                $ = q, q = $.parentNode;
              }
              $ = Y;
            }
            u = p === -1 || S === -1 ? null : { start: p, end: S };
          } else u = null;
        }
      u = u || { start: 0, end: 0 };
    } else u = null;
    for (Us = { focusedElem: l, selectionRange: u }, ws = !1, dl = n; dl !== null; )
      if (n = dl, l = n.child, (n.subtreeFlags & 1024) !== 0 && l !== null)
        l.return = n, dl = l;
      else
        for (; dl !== null; ) {
          switch (n = dl, r = n.alternate, l = n.flags, n.tag) {
            case 0:
              break;
            case 11:
            case 15:
              break;
            case 1:
              if ((l & 1024) !== 0 && r !== null) {
                l = void 0, u = n, s = r.memoizedProps, r = r.memoizedState, c = u.stateNode;
                try {
                  var Ae = Ei(
                    u.type,
                    s,
                    u.elementType === u.type
                  );
                  l = c.getSnapshotBeforeUpdate(
                    Ae,
                    r
                  ), c.__reactInternalSnapshotBeforeUpdate = l;
                } catch (Re) {
                  Tt(
                    u,
                    u.return,
                    Re
                  );
                }
              }
              break;
            case 3:
              if ((l & 1024) !== 0) {
                if (l = n.stateNode.containerInfo, u = l.nodeType, u === 9)
                  Cs(l);
                else if (u === 1)
                  switch (l.nodeName) {
                    case "HEAD":
                    case "HTML":
                    case "BODY":
                      Cs(l);
                      break;
                    default:
                      l.textContent = "";
                  }
              }
              break;
            case 5:
            case 26:
            case 27:
            case 6:
            case 4:
            case 17:
              break;
            default:
              if ((l & 1024) !== 0) throw Error(_(163));
          }
          if (l = n.sibling, l !== null) {
            l.return = n.return, dl = l;
            break;
          }
          dl = n.return;
        }
  }
  function Wy(l, n, u) {
    var c = u.flags;
    switch (u.tag) {
      case 0:
      case 11:
      case 15:
        In(l, u), c & 4 && ps(5, u);
        break;
      case 1:
        if (In(l, u), c & 4)
          if (l = u.stateNode, n === null)
            try {
              l.componentDidMount();
            } catch (y) {
              Tt(u, u.return, y);
            }
          else {
            var s = Ei(
              u.type,
              n.memoizedProps
            );
            n = n.memoizedState;
            try {
              l.componentDidUpdate(
                s,
                n,
                l.__reactInternalSnapshotBeforeUpdate
              );
            } catch (y) {
              Tt(
                u,
                u.return,
                y
              );
            }
          }
        c & 64 && Sd(u), c & 512 && No(u, u.return);
        break;
      case 3:
        if (In(l, u), c & 64 && (l = u.updateQueue, l !== null)) {
          if (n = null, u.child !== null)
            switch (u.child.tag) {
              case 27:
              case 5:
                n = u.child.stateNode;
                break;
              case 1:
                n = u.child.stateNode;
            }
          try {
            Wf(l, n);
          } catch (y) {
            Tt(u, u.return, y);
          }
        }
        break;
      case 27:
        n === null && c & 4 && Ed(u);
      case 26:
      case 5:
        In(l, u), n === null && c & 4 && wo(u), c & 512 && No(u, u.return);
        break;
      case 12:
        In(l, u);
        break;
      case 13:
        In(l, u), c & 4 && Od(l, u), c & 64 && (l = u.memoizedState, l !== null && (l = l.dehydrated, l !== null && (u = Cg.bind(
          null,
          u
        ), Yg(l, u))));
        break;
      case 22:
        if (c = u.memoizedState !== null || Wn, !c) {
          n = n !== null && n.memoizedState !== null || Jt, s = Wn;
          var r = Jt;
          Wn = c, (Jt = n) && !r ? Oi(
            l,
            u,
            (u.subtreeFlags & 8772) !== 0
          ) : In(l, u), Wn = s, Jt = r;
        }
        break;
      case 30:
        break;
      default:
        In(l, u);
    }
  }
  function Fy(l) {
    var n = l.alternate;
    n !== null && (l.alternate = null, Fy(n)), l.child = null, l.deletions = null, l.sibling = null, l.tag === 5 && (n = l.stateNode, n !== null && Ef(n)), l.stateNode = null, l.return = null, l.dependencies = null, l.memoizedProps = null, l.memoizedState = null, l.pendingProps = null, l.stateNode = null, l.updateQueue = null;
  }
  var qt = null, Cl = !1;
  function Fn(l, n, u) {
    for (u = u.child; u !== null; )
      Pe(l, n, u), u = u.sibling;
  }
  function Pe(l, n, u) {
    if (Ol && typeof Ol.onCommitFiberUnmount == "function")
      try {
        Ol.onCommitFiberUnmount(Pu, u);
      } catch {
      }
    switch (u.tag) {
      case 26:
        Jt || Sn(u, n), Fn(
          l,
          n,
          u
        ), u.memoizedState ? u.memoizedState.count-- : u.stateNode && (u = u.stateNode, u.parentNode.removeChild(u));
        break;
      case 27:
        Jt || Sn(u, n);
        var c = qt, s = Cl;
        xi(u.type) && (qt = u.stateNode, Cl = !1), Fn(
          l,
          n,
          u
        ), va(u.stateNode), qt = c, Cl = s;
        break;
      case 5:
        Jt || Sn(u, n);
      case 6:
        if (c = qt, s = Cl, qt = null, Fn(
          l,
          n,
          u
        ), qt = c, Cl = s, qt !== null)
          if (Cl)
            try {
              (qt.nodeType === 9 ? qt.body : qt.nodeName === "HTML" ? qt.ownerDocument.body : qt).removeChild(u.stateNode);
            } catch (r) {
              Tt(
                u,
                n,
                r
              );
            }
          else
            try {
              qt.removeChild(u.stateNode);
            } catch (r) {
              Tt(
                u,
                n,
                r
              );
            }
        break;
      case 18:
        qt !== null && (Cl ? (l = qt, Vd(
          l.nodeType === 9 ? l.body : l.nodeName === "HTML" ? l.ownerDocument.body : l,
          u.stateNode
        ), iu(l)) : Vd(qt, u.stateNode));
        break;
      case 4:
        c = qt, s = Cl, qt = u.stateNode.containerInfo, Cl = !0, Fn(
          l,
          n,
          u
        ), qt = c, Cl = s;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        Jt || Ri(2, u, n), Jt || Ri(4, u, n), Fn(
          l,
          n,
          u
        );
        break;
      case 1:
        Jt || (Sn(u, n), c = u.stateNode, typeof c.componentWillUnmount == "function" && Jy(
          u,
          n,
          c
        )), Fn(
          l,
          n,
          u
        );
        break;
      case 21:
        Fn(
          l,
          n,
          u
        );
        break;
      case 22:
        Jt = (c = Jt) || u.memoizedState !== null, Fn(
          l,
          n,
          u
        ), Jt = c;
        break;
      default:
        Fn(
          l,
          n,
          u
        );
    }
  }
  function Od(l, n) {
    if (n.memoizedState === null && (l = n.alternate, l !== null && (l = l.memoizedState, l !== null && (l = l.dehydrated, l !== null))))
      try {
        iu(l);
      } catch (u) {
        Tt(n, n.return, u);
      }
  }
  function Iy(l) {
    switch (l.tag) {
      case 13:
      case 19:
        var n = l.stateNode;
        return n === null && (n = l.stateNode = new Rd()), n;
      case 22:
        return l = l.stateNode, n = l._retryCache, n === null && (n = l._retryCache = new Rd()), n;
      default:
        throw Error(_(435, l.tag));
    }
  }
  function Dd(l, n) {
    var u = Iy(l);
    n.forEach(function(c) {
      var s = xg.bind(null, l, c);
      u.has(c) || (u.add(c), c.then(s, s));
    });
  }
  function Fl(l, n) {
    var u = n.deletions;
    if (u !== null)
      for (var c = 0; c < u.length; c++) {
        var s = u[c], r = l, y = n, p = y;
        e: for (; p !== null; ) {
          switch (p.tag) {
            case 27:
              if (xi(p.type)) {
                qt = p.stateNode, Cl = !1;
                break e;
              }
              break;
            case 5:
              qt = p.stateNode, Cl = !1;
              break e;
            case 3:
            case 4:
              qt = p.stateNode.containerInfo, Cl = !0;
              break e;
          }
          p = p.return;
        }
        if (qt === null) throw Error(_(160));
        Pe(r, y, s), qt = null, Cl = !1, r = s.alternate, r !== null && (r.return = null), s.return = null;
      }
    if (n.subtreeFlags & 13878)
      for (n = n.child; n !== null; )
        vs(n, l), n = n.sibling;
  }
  var Il = null;
  function vs(l, n) {
    var u = l.alternate, c = l.flags;
    switch (l.tag) {
      case 0:
      case 11:
      case 14:
      case 15:
        Fl(n, l), Tl(l), c & 4 && (Ri(3, l, l.return), ps(3, l), Ri(5, l, l.return));
        break;
      case 1:
        Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), c & 64 && Wn && (l = l.updateQueue, l !== null && (c = l.callbacks, c !== null && (u = l.shared.hiddenCallbacks, l.shared.hiddenCallbacks = u === null ? c : u.concat(c))));
        break;
      case 26:
        var s = Il;
        if (Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), c & 4) {
          var r = u !== null ? u.memoizedState : null;
          if (c = l.memoizedState, u === null)
            if (c === null)
              if (l.stateNode === null) {
                e: {
                  c = l.type, u = l.memoizedProps, s = s.ownerDocument || s;
                  t: switch (c) {
                    case "title":
                      r = s.getElementsByTagName("title")[0], (!r || r[ye] || r[vl] || r.namespaceURI === "http://www.w3.org/2000/svg" || r.hasAttribute("itemprop")) && (r = s.createElement(c), s.head.insertBefore(
                        r,
                        s.querySelector("head > title")
                      )), Ce(r, c, u), r[vl] = l, ol(r), c = r;
                      break e;
                    case "link":
                      var y = Om(
                        "link",
                        "href",
                        s
                      ).get(c + (u.href || ""));
                      if (y) {
                        for (var p = 0; p < y.length; p++)
                          if (r = y[p], r.getAttribute("href") === (u.href == null || u.href === "" ? null : u.href) && r.getAttribute("rel") === (u.rel == null ? null : u.rel) && r.getAttribute("title") === (u.title == null ? null : u.title) && r.getAttribute("crossorigin") === (u.crossOrigin == null ? null : u.crossOrigin)) {
                            y.splice(p, 1);
                            break t;
                          }
                      }
                      r = s.createElement(c), Ce(r, c, u), s.head.appendChild(r);
                      break;
                    case "meta":
                      if (y = Om(
                        "meta",
                        "content",
                        s
                      ).get(c + (u.content || ""))) {
                        for (p = 0; p < y.length; p++)
                          if (r = y[p], r.getAttribute("content") === (u.content == null ? null : "" + u.content) && r.getAttribute("name") === (u.name == null ? null : u.name) && r.getAttribute("property") === (u.property == null ? null : u.property) && r.getAttribute("http-equiv") === (u.httpEquiv == null ? null : u.httpEquiv) && r.getAttribute("charset") === (u.charSet == null ? null : u.charSet)) {
                            y.splice(p, 1);
                            break t;
                          }
                      }
                      r = s.createElement(c), Ce(r, c, u), s.head.appendChild(r);
                      break;
                    default:
                      throw Error(_(468, c));
                  }
                  r[vl] = l, ol(r), c = r;
                }
                l.stateNode = c;
              } else
                Dm(
                  s,
                  l.type,
                  l.stateNode
                );
            else
              l.stateNode = Dv(
                s,
                c,
                l.memoizedProps
              );
          else
            r !== c ? (r === null ? u.stateNode !== null && (u = u.stateNode, u.parentNode.removeChild(u)) : r.count--, c === null ? Dm(
              s,
              l.type,
              l.stateNode
            ) : Dv(
              s,
              c,
              l.memoizedProps
            )) : c === null && l.stateNode !== null && ky(
              l,
              l.memoizedProps,
              u.memoizedProps
            );
        }
        break;
      case 27:
        Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), u !== null && c & 4 && ky(
          l,
          l.memoizedProps,
          u.memoizedProps
        );
        break;
      case 5:
        if (Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), l.flags & 32) {
          s = l.stateNode;
          try {
            uo(s, "");
          } catch (Y) {
            Tt(l, l.return, Y);
          }
        }
        c & 4 && l.stateNode != null && (s = l.memoizedProps, ky(
          l,
          s,
          u !== null ? u.memoizedProps : s
        )), c & 1024 && (Ad = !0);
        break;
      case 6:
        if (Fl(n, l), Tl(l), c & 4) {
          if (l.stateNode === null)
            throw Error(_(162));
          c = l.memoizedProps, u = l.stateNode;
          try {
            u.nodeValue = c;
          } catch (Y) {
            Tt(l, l.return, Y);
          }
        }
        break;
      case 3:
        if (qi = null, s = Il, Il = Xd(n.containerInfo), Fl(n, l), Il = s, Tl(l), c & 4 && u !== null && u.memoizedState.isDehydrated)
          try {
            iu(n.containerInfo);
          } catch (Y) {
            Tt(l, l.return, Y);
          }
        Ad && (Ad = !1, Py(l));
        break;
      case 4:
        c = Il, Il = Xd(
          l.stateNode.containerInfo
        ), Fl(n, l), Tl(l), Il = c;
        break;
      case 12:
        Fl(n, l), Tl(l);
        break;
      case 13:
        Fl(n, l), Tl(l), l.child.flags & 8192 && l.memoizedState !== null != (u !== null && u.memoizedState !== null) && (cm = pl()), c & 4 && (c = l.updateQueue, c !== null && (l.updateQueue = null, Dd(l, c)));
        break;
      case 22:
        s = l.memoizedState !== null;
        var S = u !== null && u.memoizedState !== null, H = Wn, K = Jt;
        if (Wn = H || s, Jt = K || S, Fl(n, l), Jt = K, Wn = H, Tl(l), c & 8192)
          e: for (n = l.stateNode, n._visibility = s ? n._visibility & -2 : n._visibility | 1, s && (u === null || S || Wn || Jt || jt(l)), u = null, n = l; ; ) {
            if (n.tag === 5 || n.tag === 26) {
              if (u === null) {
                S = u = n;
                try {
                  if (r = S.stateNode, s)
                    y = r.style, typeof y.setProperty == "function" ? y.setProperty("display", "none", "important") : y.display = "none";
                  else {
                    p = S.stateNode;
                    var $ = S.memoizedProps.style, q = $ != null && $.hasOwnProperty("display") ? $.display : null;
                    p.style.display = q == null || typeof q == "boolean" ? "" : ("" + q).trim();
                  }
                } catch (Y) {
                  Tt(S, S.return, Y);
                }
              }
            } else if (n.tag === 6) {
              if (u === null) {
                S = n;
                try {
                  S.stateNode.nodeValue = s ? "" : S.memoizedProps;
                } catch (Y) {
                  Tt(S, S.return, Y);
                }
              }
            } else if ((n.tag !== 22 && n.tag !== 23 || n.memoizedState === null || n === l) && n.child !== null) {
              n.child.return = n, n = n.child;
              continue;
            }
            if (n === l) break e;
            for (; n.sibling === null; ) {
              if (n.return === null || n.return === l) break e;
              u === n && (u = null), n = n.return;
            }
            u === n && (u = null), n.sibling.return = n.return, n = n.sibling;
          }
        c & 4 && (c = l.updateQueue, c !== null && (u = c.retryQueue, u !== null && (c.retryQueue = null, Dd(l, u))));
        break;
      case 19:
        Fl(n, l), Tl(l), c & 4 && (c = l.updateQueue, c !== null && (l.updateQueue = null, Dd(l, c)));
        break;
      case 30:
        break;
      case 21:
        break;
      default:
        Fl(n, l), Tl(l);
    }
  }
  function Tl(l) {
    var n = l.flags;
    if (n & 2) {
      try {
        for (var u, c = l.return; c !== null; ) {
          if (fv(c)) {
            u = c;
            break;
          }
          c = c.return;
        }
        if (u == null) throw Error(_(160));
        switch (u.tag) {
          case 27:
            var s = u.stateNode, r = $a(l);
            Td(l, r, s);
            break;
          case 5:
            var y = u.stateNode;
            u.flags & 32 && (uo(y, ""), u.flags &= -33);
            var p = $a(l);
            Td(l, p, y);
            break;
          case 3:
          case 4:
            var S = u.stateNode.containerInfo, H = $a(l);
            Dc(
              l,
              H,
              S
            );
            break;
          default:
            throw Error(_(161));
        }
      } catch (K) {
        Tt(l, l.return, K);
      }
      l.flags &= -3;
    }
    n & 4096 && (l.flags &= -4097);
  }
  function Py(l) {
    if (l.subtreeFlags & 1024)
      for (l = l.child; l !== null; ) {
        var n = l;
        Py(n), n.tag === 5 && n.flags & 1024 && n.stateNode.reset(), l = l.sibling;
      }
  }
  function In(l, n) {
    if (n.subtreeFlags & 8772)
      for (n = n.child; n !== null; )
        Wy(l, n.alternate, n), n = n.sibling;
  }
  function jt(l) {
    for (l = l.child; l !== null; ) {
      var n = l;
      switch (n.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          Ri(4, n, n.return), jt(n);
          break;
        case 1:
          Sn(n, n.return);
          var u = n.stateNode;
          typeof u.componentWillUnmount == "function" && Jy(
            n,
            n.return,
            u
          ), jt(n);
          break;
        case 27:
          va(n.stateNode);
        case 26:
        case 5:
          Sn(n, n.return), jt(n);
          break;
        case 22:
          n.memoizedState === null && jt(n);
          break;
        case 30:
          jt(n);
          break;
        default:
          jt(n);
      }
      l = l.sibling;
    }
  }
  function Oi(l, n, u) {
    for (u = u && (n.subtreeFlags & 8772) !== 0, n = n.child; n !== null; ) {
      var c = n.alternate, s = l, r = n, y = r.flags;
      switch (r.tag) {
        case 0:
        case 11:
        case 15:
          Oi(
            s,
            r,
            u
          ), ps(4, r);
          break;
        case 1:
          if (Oi(
            s,
            r,
            u
          ), c = r, s = c.stateNode, typeof s.componentDidMount == "function")
            try {
              s.componentDidMount();
            } catch (H) {
              Tt(c, c.return, H);
            }
          if (c = r, s = c.updateQueue, s !== null) {
            var p = c.stateNode;
            try {
              var S = s.shared.hiddenCallbacks;
              if (S !== null)
                for (s.shared.hiddenCallbacks = null, s = 0; s < S.length; s++)
                  Vr(S[s], p);
            } catch (H) {
              Tt(c, c.return, H);
            }
          }
          u && y & 64 && Sd(r), No(r, r.return);
          break;
        case 27:
          Ed(r);
        case 26:
        case 5:
          Oi(
            s,
            r,
            u
          ), u && c === null && y & 4 && wo(r), No(r, r.return);
          break;
        case 12:
          Oi(
            s,
            r,
            u
          );
          break;
        case 13:
          Oi(
            s,
            r,
            u
          ), u && y & 4 && Od(s, r);
          break;
        case 22:
          r.memoizedState === null && Oi(
            s,
            r,
            u
          ), No(r, r.return);
          break;
        case 30:
          break;
        default:
          Oi(
            s,
            r,
            u
          );
      }
      n = n.sibling;
    }
  }
  function Wa(l, n) {
    var u = null;
    l !== null && l.memoizedState !== null && l.memoizedState.cachePool !== null && (u = l.memoizedState.cachePool.pool), l = null, n.memoizedState !== null && n.memoizedState.cachePool !== null && (l = n.memoizedState.cachePool.pool), l !== u && (l != null && l.refCount++, u != null && Yn(u));
  }
  function zd(l, n) {
    l = null, n.alternate !== null && (l = n.alternate.memoizedState.cache), n = n.memoizedState.cache, n !== l && (n.refCount++, l != null && Yn(l));
  }
  function xl(l, n, u, c) {
    if (n.subtreeFlags & 10256)
      for (n = n.child; n !== null; )
        em(
          l,
          n,
          u,
          c
        ), n = n.sibling;
  }
  function em(l, n, u, c) {
    var s = n.flags;
    switch (n.tag) {
      case 0:
      case 11:
      case 15:
        xl(
          l,
          n,
          u,
          c
        ), s & 2048 && ps(9, n);
        break;
      case 1:
        xl(
          l,
          n,
          u,
          c
        );
        break;
      case 3:
        xl(
          l,
          n,
          u,
          c
        ), s & 2048 && (l = null, n.alternate !== null && (l = n.alternate.memoizedState.cache), n = n.memoizedState.cache, n !== l && (n.refCount++, l != null && Yn(l)));
        break;
      case 12:
        if (s & 2048) {
          xl(
            l,
            n,
            u,
            c
          ), l = n.stateNode;
          try {
            var r = n.memoizedProps, y = r.id, p = r.onPostCommit;
            typeof p == "function" && p(
              y,
              n.alternate === null ? "mount" : "update",
              l.passiveEffectDuration,
              -0
            );
          } catch (S) {
            Tt(n, n.return, S);
          }
        } else
          xl(
            l,
            n,
            u,
            c
          );
        break;
      case 13:
        xl(
          l,
          n,
          u,
          c
        );
        break;
      case 23:
        break;
      case 22:
        r = n.stateNode, y = n.alternate, n.memoizedState !== null ? r._visibility & 2 ? xl(
          l,
          n,
          u,
          c
        ) : ht(l, n) : r._visibility & 2 ? xl(
          l,
          n,
          u,
          c
        ) : (r._visibility |= 2, wu(
          l,
          n,
          u,
          c,
          (n.subtreeFlags & 10256) !== 0
        )), s & 2048 && Wa(y, n);
        break;
      case 24:
        xl(
          l,
          n,
          u,
          c
        ), s & 2048 && zd(n.alternate, n);
        break;
      default:
        xl(
          l,
          n,
          u,
          c
        );
    }
  }
  function wu(l, n, u, c, s) {
    for (s = s && (n.subtreeFlags & 10256) !== 0, n = n.child; n !== null; ) {
      var r = l, y = n, p = u, S = c, H = y.flags;
      switch (y.tag) {
        case 0:
        case 11:
        case 15:
          wu(
            r,
            y,
            p,
            S,
            s
          ), ps(8, y);
          break;
        case 23:
          break;
        case 22:
          var K = y.stateNode;
          y.memoizedState !== null ? K._visibility & 2 ? wu(
            r,
            y,
            p,
            S,
            s
          ) : ht(
            r,
            y
          ) : (K._visibility |= 2, wu(
            r,
            y,
            p,
            S,
            s
          )), s && H & 2048 && Wa(
            y.alternate,
            y
          );
          break;
        case 24:
          wu(
            r,
            y,
            p,
            S,
            s
          ), s && H & 2048 && zd(y.alternate, y);
          break;
        default:
          wu(
            r,
            y,
            p,
            S,
            s
          );
      }
      n = n.sibling;
    }
  }
  function ht(l, n) {
    if (n.subtreeFlags & 10256)
      for (n = n.child; n !== null; ) {
        var u = l, c = n, s = c.flags;
        switch (c.tag) {
          case 22:
            ht(u, c), s & 2048 && Wa(
              c.alternate,
              c
            );
            break;
          case 24:
            ht(u, c), s & 2048 && zd(c.alternate, c);
            break;
          default:
            ht(u, c);
        }
        n = n.sibling;
      }
  }
  var zc = 8192;
  function kt(l) {
    if (l.subtreeFlags & zc)
      for (l = l.child; l !== null; )
        sv(l), l = l.sibling;
  }
  function sv(l) {
    switch (l.tag) {
      case 26:
        kt(l), l.flags & zc && l.memoizedState !== null && _v(
          Il,
          l.memoizedState,
          l.memoizedProps
        );
        break;
      case 5:
        kt(l);
        break;
      case 3:
      case 4:
        var n = Il;
        Il = Xd(l.stateNode.containerInfo), kt(l), Il = n;
        break;
      case 22:
        l.memoizedState === null && (n = l.alternate, n !== null && n.memoizedState !== null ? (n = zc, zc = 16777216, kt(l), zc = n) : kt(l));
        break;
      default:
        kt(l);
    }
  }
  function tm(l) {
    var n = l.alternate;
    if (n !== null && (l = n.child, l !== null)) {
      n.child = null;
      do
        n = l.sibling, l.sibling = null, l = n;
      while (l !== null);
    }
  }
  function Mc(l) {
    var n = l.deletions;
    if ((l.flags & 16) !== 0) {
      if (n !== null)
        for (var u = 0; u < n.length; u++) {
          var c = n[u];
          dl = c, am(
            c,
            l
          );
        }
      tm(l);
    }
    if (l.subtreeFlags & 10256)
      for (l = l.child; l !== null; )
        lm(l), l = l.sibling;
  }
  function lm(l) {
    switch (l.tag) {
      case 0:
      case 11:
      case 15:
        Mc(l), l.flags & 2048 && Ri(9, l, l.return);
        break;
      case 3:
        Mc(l);
        break;
      case 12:
        Mc(l);
        break;
      case 22:
        var n = l.stateNode;
        l.memoizedState !== null && n._visibility & 2 && (l.return === null || l.return.tag !== 13) ? (n._visibility &= -3, Pl(l)) : Mc(l);
        break;
      default:
        Mc(l);
    }
  }
  function Pl(l) {
    var n = l.deletions;
    if ((l.flags & 16) !== 0) {
      if (n !== null)
        for (var u = 0; u < n.length; u++) {
          var c = n[u];
          dl = c, am(
            c,
            l
          );
        }
      tm(l);
    }
    for (l = l.child; l !== null; ) {
      switch (n = l, n.tag) {
        case 0:
        case 11:
        case 15:
          Ri(8, n, n.return), Pl(n);
          break;
        case 22:
          u = n.stateNode, u._visibility & 2 && (u._visibility &= -3, Pl(n));
          break;
        default:
          Pl(n);
      }
      l = l.sibling;
    }
  }
  function am(l, n) {
    for (; dl !== null; ) {
      var u = dl;
      switch (u.tag) {
        case 0:
        case 11:
        case 15:
          Ri(8, u, n);
          break;
        case 23:
        case 22:
          if (u.memoizedState !== null && u.memoizedState.cachePool !== null) {
            var c = u.memoizedState.cachePool.pool;
            c != null && c.refCount++;
          }
          break;
        case 24:
          Yn(u.memoizedState.cache);
      }
      if (c = u.child, c !== null) c.return = u, dl = c;
      else
        e: for (u = l; dl !== null; ) {
          c = dl;
          var s = c.sibling, r = c.return;
          if (Fy(c), c === u) {
            dl = null;
            break e;
          }
          if (s !== null) {
            s.return = r, dl = s;
            break e;
          }
          dl = r;
        }
    }
  }
  var nm = {
    getCacheForType: function(l) {
      var n = gl(fl), u = n.data.get(l);
      return u === void 0 && (u = l(), n.data.set(l, u)), u;
    }
  }, rv = typeof WeakMap == "function" ? WeakMap : Map, gt = 0, _t = null, tt = null, lt = 0, St = 0, ya = null, Pn = !1, qo = !1, um = !1, qu = 0, $t = 0, ju = 0, _c = 0, eu = 0, Fa = 0, jo = 0, Bo = null, ma = null, im = !1, cm = 0, Md = 1 / 0, Yo = null, Di = null, Hl = 0, tu = null, Go = null, Nl = 0, _d = 0, Ud = null, om = null, Lo = 0, fm = null;
  function _a() {
    if ((gt & 2) !== 0 && lt !== 0)
      return lt & -lt;
    if (O.T !== null) {
      var l = Ka;
      return l !== 0 ? l : Hc();
    }
    return or();
  }
  function sm() {
    Fa === 0 && (Fa = (lt & 536870912) === 0 || rt ? le() : 536870912);
    var l = Ma.current;
    return l !== null && (l.flags |= 32), Fa;
  }
  function Ua(l, n, u) {
    (l === _t && (St === 2 || St === 9) || l.cancelPendingCommit !== null) && (lu(l, 0), Bu(
      l,
      lt,
      Fa,
      !1
    )), Ne(l, u), ((gt & 2) === 0 || l !== _t) && (l === _t && ((gt & 2) === 0 && (_c |= u), $t === 4 && Bu(
      l,
      lt,
      Fa,
      !1
    )), pa(l));
  }
  function Vo(l, n, u) {
    if ((gt & 6) !== 0) throw Error(_(327));
    var c = !u && (n & 124) === 0 && (n & l.expiredLanes) === 0 || m(l, n), s = c ? dm(l, n) : Cd(l, n, !0), r = c;
    do {
      if (s === 0) {
        qo && !c && Bu(l, n, 0, !1);
        break;
      } else {
        if (u = l.current.alternate, r && !dv(u)) {
          s = Cd(l, n, !1), r = !1;
          continue;
        }
        if (s === 2) {
          if (r = n, l.errorRecoveryDisabledLanes & r)
            var y = 0;
          else
            y = l.pendingLanes & -536870913, y = y !== 0 ? y : y & 536870912 ? 536870912 : 0;
          if (y !== 0) {
            n = y;
            e: {
              var p = l;
              s = Bo;
              var S = p.current.memoizedState.isDehydrated;
              if (S && (lu(p, y).flags |= 256), y = Cd(
                p,
                y,
                !1
              ), y !== 2) {
                if (um && !S) {
                  p.errorRecoveryDisabledLanes |= r, _c |= r, s = 4;
                  break e;
                }
                r = ma, ma = s, r !== null && (ma === null ? ma = r : ma.push.apply(
                  ma,
                  r
                ));
              }
              s = y;
            }
            if (r = !1, s !== 2) continue;
          }
        }
        if (s === 1) {
          lu(l, 0), Bu(l, n, 0, !0);
          break;
        }
        e: {
          switch (c = l, r = s, r) {
            case 0:
            case 1:
              throw Error(_(345));
            case 4:
              if ((n & 4194048) !== n) break;
            case 6:
              Bu(
                c,
                n,
                Fa,
                !Pn
              );
              break e;
            case 2:
              ma = null;
              break;
            case 3:
            case 5:
              break;
            default:
              throw Error(_(329));
          }
          if ((n & 62914560) === n && (s = cm + 300 - pl(), 10 < s)) {
            if (Bu(
              c,
              n,
              Fa,
              !Pn
            ), un(c, 0, !0) !== 0) break e;
            c.timeoutHandle = Ld(
              gs.bind(
                null,
                c,
                u,
                ma,
                Yo,
                im,
                n,
                Fa,
                _c,
                jo,
                Pn,
                r,
                2,
                -0,
                0
              ),
              s
            );
            break e;
          }
          gs(
            c,
            u,
            ma,
            Yo,
            im,
            n,
            Fa,
            _c,
            jo,
            Pn,
            r,
            0,
            -0,
            0
          );
        }
      }
      break;
    } while (!0);
    pa(l);
  }
  function gs(l, n, u, c, s, r, y, p, S, H, K, $, q, Y) {
    if (l.timeoutHandle = -1, $ = n.subtreeFlags, ($ & 8192 || ($ & 16785408) === 16785408) && (ef = { stylesheets: null, count: 0, unsuspend: Mv }, sv(n), $ = Mm(), $ !== null)) {
      l.cancelPendingCommit = $(
        mv.bind(
          null,
          l,
          n,
          r,
          u,
          c,
          s,
          y,
          p,
          S,
          K,
          1,
          q,
          Y
        )
      ), Bu(l, r, y, !H);
      return;
    }
    mv(
      l,
      n,
      r,
      u,
      c,
      s,
      y,
      p,
      S
    );
  }
  function dv(l) {
    for (var n = l; ; ) {
      var u = n.tag;
      if ((u === 0 || u === 11 || u === 15) && n.flags & 16384 && (u = n.updateQueue, u !== null && (u = u.stores, u !== null)))
        for (var c = 0; c < u.length; c++) {
          var s = u[c], r = s.getSnapshot;
          s = s.value;
          try {
            if (!Ul(r(), s)) return !1;
          } catch {
            return !1;
          }
        }
      if (u = n.child, n.subtreeFlags & 16384 && u !== null)
        u.return = n, n = u;
      else {
        if (n === l) break;
        for (; n.sibling === null; ) {
          if (n.return === null || n.return === l) return !0;
          n = n.return;
        }
        n.sibling.return = n.return, n = n.sibling;
      }
    }
    return !0;
  }
  function Bu(l, n, u, c) {
    n &= ~eu, n &= ~_c, l.suspendedLanes |= n, l.pingedLanes &= ~n, c && (l.warmLanes |= n), c = l.expirationTimes;
    for (var s = n; 0 < s; ) {
      var r = 31 - Dl(s), y = 1 << r;
      c[r] = -1, s &= ~y;
    }
    u !== 0 && ct(l, u, n);
  }
  function Uc() {
    return (gt & 6) === 0 ? (Es(0), !1) : !0;
  }
  function zi() {
    if (tt !== null) {
      if (St === 0)
        var l = tt.return;
      else
        l = tt, pn = Mu = null, Kr(l), Sc = null, _o = 0, l = tt;
      for (; l !== null; )
        Ky(l.alternate, l), l = l.return;
      tt = null;
    }
  }
  function lu(l, n) {
    var u = l.timeoutHandle;
    u !== -1 && (l.timeoutHandle = -1, qg(u)), u = l.cancelPendingCommit, u !== null && (l.cancelPendingCommit = null, u()), zi(), _t = l, tt = u = yn(l.current, null), lt = n, St = 0, ya = null, Pn = !1, qo = m(l, n), um = !1, jo = Fa = eu = _c = ju = $t = 0, ma = Bo = null, im = !1, (n & 8) !== 0 && (n |= n & 32);
    var c = l.entangledLanes;
    if (c !== 0)
      for (l = l.entanglements, c &= n; 0 < c; ) {
        var s = 31 - Dl(c), r = 1 << s;
        n |= l[s], c &= ~r;
      }
    return qu = n, hn(), u;
  }
  function rm(l, n) {
    Ge = null, O.H = cd, n === vi || n === kf ? (n = my(), St = 3) : n === jr ? (n = my(), St = 4) : St = n === Kt ? 8 : n !== null && typeof n == "object" && typeof n.then == "function" ? 6 : 1, ya = n, tt === null && ($t = 1, ds(
      l,
      Oa(n, l.current)
    ));
  }
  function hv() {
    var l = O.H;
    return O.H = cd, l === null ? cd : l;
  }
  function Cc() {
    var l = O.A;
    return O.A = nm, l;
  }
  function xc() {
    $t = 4, Pn || (lt & 4194048) !== lt && Ma.current !== null || (qo = !0), (ju & 134217727) === 0 && (_c & 134217727) === 0 || _t === null || Bu(
      _t,
      lt,
      Fa,
      !1
    );
  }
  function Cd(l, n, u) {
    var c = gt;
    gt |= 2;
    var s = hv(), r = Cc();
    (_t !== l || lt !== n) && (Yo = null, lu(l, n)), n = !1;
    var y = $t;
    e: do
      try {
        if (St !== 0 && tt !== null) {
          var p = tt, S = ya;
          switch (St) {
            case 8:
              zi(), y = 6;
              break e;
            case 3:
            case 2:
            case 9:
            case 6:
              Ma.current === null && (n = !0);
              var H = St;
              if (St = 0, ya = null, Xo(l, p, S, H), u && qo) {
                y = 0;
                break e;
              }
              break;
            default:
              H = St, St = 0, ya = null, Xo(l, p, S, H);
          }
        }
        xd(), y = $t;
        break;
      } catch (K) {
        rm(l, K);
      }
    while (!0);
    return n && l.shellSuspendCounter++, pn = Mu = null, gt = c, O.H = s, O.A = r, tt === null && (_t = null, lt = 0, hn()), y;
  }
  function xd() {
    for (; tt !== null; ) ym(tt);
  }
  function dm(l, n) {
    var u = gt;
    gt |= 2;
    var c = hv(), s = Cc();
    _t !== l || lt !== n ? (Yo = null, Md = pl() + 500, lu(l, n)) : qo = m(
      l,
      n
    );
    e: do
      try {
        if (St !== 0 && tt !== null) {
          n = tt;
          var r = ya;
          t: switch (St) {
            case 1:
              St = 0, ya = null, Xo(l, n, r, 1);
              break;
            case 2:
            case 9:
              if (Yr(r)) {
                St = 0, ya = null, mm(n);
                break;
              }
              n = function() {
                St !== 2 && St !== 9 || _t !== l || (St = 7), pa(l);
              }, r.then(n, n);
              break e;
            case 3:
              St = 7;
              break e;
            case 4:
              St = 5;
              break e;
            case 7:
              Yr(r) ? (St = 0, ya = null, mm(n)) : (St = 0, ya = null, Xo(l, n, r, 7));
              break;
            case 5:
              var y = null;
              switch (tt.tag) {
                case 26:
                  y = tt.memoizedState;
                case 5:
                case 27:
                  var p = tt;
                  if (!y || zm(y)) {
                    St = 0, ya = null;
                    var S = p.sibling;
                    if (S !== null) tt = S;
                    else {
                      var H = p.return;
                      H !== null ? (tt = H, bs(H)) : tt = null;
                    }
                    break t;
                  }
              }
              St = 0, ya = null, Xo(l, n, r, 5);
              break;
            case 6:
              St = 0, ya = null, Xo(l, n, r, 6);
              break;
            case 8:
              zi(), $t = 6;
              break e;
            default:
              throw Error(_(462));
          }
        }
        hm();
        break;
      } catch (K) {
        rm(l, K);
      }
    while (!0);
    return pn = Mu = null, O.H = c, O.A = s, gt = u, tt !== null ? 0 : (_t = null, lt = 0, hn(), $t);
  }
  function hm() {
    for (; tt !== null && !bf(); )
      ym(tt);
  }
  function ym(l) {
    var n = ov(l.alternate, l, qu);
    l.memoizedProps = l.pendingProps, n === null ? bs(l) : tt = n;
  }
  function mm(l) {
    var n = l, u = n.alternate;
    switch (n.tag) {
      case 15:
      case 0:
        n = Ly(
          u,
          n,
          n.pendingProps,
          n.type,
          void 0,
          lt
        );
        break;
      case 11:
        n = Ly(
          u,
          n,
          n.pendingProps,
          n.type.render,
          n.ref,
          lt
        );
        break;
      case 5:
        Kr(n);
      default:
        Ky(u, n), n = tt = $e(n, qu), n = ov(u, n, qu);
    }
    l.memoizedProps = l.pendingProps, n === null ? bs(l) : tt = n;
  }
  function Xo(l, n, u, c) {
    pn = Mu = null, Kr(n), Sc = null, _o = 0;
    var s = n.return;
    try {
      if (uv(
        l,
        s,
        n,
        u,
        lt
      )) {
        $t = 1, ds(
          l,
          Oa(u, l.current)
        ), tt = null;
        return;
      }
    } catch (r) {
      if (s !== null) throw tt = s, r;
      $t = 1, ds(
        l,
        Oa(u, l.current)
      ), tt = null;
      return;
    }
    n.flags & 32768 ? (rt || c === 1 ? l = !0 : qo || (lt & 536870912) !== 0 ? l = !1 : (Pn = l = !0, (c === 2 || c === 9 || c === 3 || c === 6) && (c = Ma.current, c !== null && c.tag === 13 && (c.flags |= 16384))), yv(n, l)) : bs(n);
  }
  function bs(l) {
    var n = l;
    do {
      if ((n.flags & 32768) !== 0) {
        yv(
          n,
          Pn
        );
        return;
      }
      l = n.return;
      var u = Zy(
        n.alternate,
        n,
        qu
      );
      if (u !== null) {
        tt = u;
        return;
      }
      if (n = n.sibling, n !== null) {
        tt = n;
        return;
      }
      tt = n = l;
    } while (n !== null);
    $t === 0 && ($t = 5);
  }
  function yv(l, n) {
    do {
      var u = Ug(l.alternate, l);
      if (u !== null) {
        u.flags &= 32767, tt = u;
        return;
      }
      if (u = l.return, u !== null && (u.flags |= 32768, u.subtreeFlags = 0, u.deletions = null), !n && (l = l.sibling, l !== null)) {
        tt = l;
        return;
      }
      tt = l = u;
    } while (l !== null);
    $t = 6, tt = null;
  }
  function mv(l, n, u, c, s, r, y, p, S) {
    l.cancelPendingCommit = null;
    do
      Nd();
    while (Hl !== 0);
    if ((gt & 6) !== 0) throw Error(_(327));
    if (n !== null) {
      if (n === l.current) throw Error(_(177));
      if (r = n.lanes | n.childLanes, r |= wn, Ye(
        l,
        u,
        r,
        y,
        p,
        S
      ), l === _t && (tt = _t = null, lt = 0), Go = n, tu = l, Nl = u, _d = r, Ud = s, om = c, (n.subtreeFlags & 10256) !== 0 || (n.flags & 10256) !== 0 ? (l.callbackNode = null, l.callbackPriority = 0, Hg(Mn, function() {
        return pm(), null;
      })) : (l.callbackNode = null, l.callbackPriority = 0), c = (n.flags & 13878) !== 0, (n.subtreeFlags & 13878) !== 0 || c) {
        c = O.T, O.T = null, s = F.p, F.p = 2, y = gt, gt |= 4;
        try {
          $y(l, n, u);
        } finally {
          gt = y, F.p = s, O.T = c;
        }
      }
      Hl = 1, pv(), Ss(), Hd();
    }
  }
  function pv() {
    if (Hl === 1) {
      Hl = 0;
      var l = tu, n = Go, u = (n.flags & 13878) !== 0;
      if ((n.subtreeFlags & 13878) !== 0 || u) {
        u = O.T, O.T = null;
        var c = F.p;
        F.p = 2;
        var s = gt;
        gt |= 4;
        try {
          vs(n, l);
          var r = Us, y = ny(l.containerInfo), p = r.focusedElem, S = r.selectionRange;
          if (y !== p && p && p.ownerDocument && jf(
            p.ownerDocument.documentElement,
            p
          )) {
            if (S !== null && Bf(p)) {
              var H = S.start, K = S.end;
              if (K === void 0 && (K = H), "selectionStart" in p)
                p.selectionStart = H, p.selectionEnd = Math.min(
                  K,
                  p.value.length
                );
              else {
                var $ = p.ownerDocument || document, q = $ && $.defaultView || window;
                if (q.getSelection) {
                  var Y = q.getSelection(), Ae = p.textContent.length, Re = Math.min(S.start, Ae), yt = S.end === void 0 ? Re : Math.min(S.end, Ae);
                  !Y.extend && Re > yt && (y = yt, yt = Re, Re = y);
                  var M = Ct(
                    p,
                    Re
                  ), R = Ct(
                    p,
                    yt
                  );
                  if (M && R && (Y.rangeCount !== 1 || Y.anchorNode !== M.node || Y.anchorOffset !== M.offset || Y.focusNode !== R.node || Y.focusOffset !== R.offset)) {
                    var C = $.createRange();
                    C.setStart(M.node, M.offset), Y.removeAllRanges(), Re > yt ? (Y.addRange(C), Y.extend(R.node, R.offset)) : (C.setEnd(R.node, R.offset), Y.addRange(C));
                  }
                }
              }
            }
            for ($ = [], Y = p; Y = Y.parentNode; )
              Y.nodeType === 1 && $.push({
                element: Y,
                left: Y.scrollLeft,
                top: Y.scrollTop
              });
            for (typeof p.focus == "function" && p.focus(), p = 0; p < $.length; p++) {
              var k = $[p];
              k.element.scrollLeft = k.left, k.element.scrollTop = k.top;
            }
          }
          ws = !!_s, Us = _s = null;
        } finally {
          gt = s, F.p = c, O.T = u;
        }
      }
      l.current = n, Hl = 2;
    }
  }
  function Ss() {
    if (Hl === 2) {
      Hl = 0;
      var l = tu, n = Go, u = (n.flags & 8772) !== 0;
      if ((n.subtreeFlags & 8772) !== 0 || u) {
        u = O.T, O.T = null;
        var c = F.p;
        F.p = 2;
        var s = gt;
        gt |= 4;
        try {
          Wy(l, n.alternate, n);
        } finally {
          gt = s, F.p = c, O.T = u;
        }
      }
      Hl = 3;
    }
  }
  function Hd() {
    if (Hl === 4 || Hl === 3) {
      Hl = 0, tl();
      var l = tu, n = Go, u = Nl, c = om;
      (n.subtreeFlags & 10256) !== 0 || (n.flags & 10256) !== 0 ? Hl = 5 : (Hl = 0, Go = tu = null, vv(l, l.pendingLanes));
      var s = l.pendingLanes;
      if (s === 0 && (Di = null), cn(u), n = n.stateNode, Ol && typeof Ol.onCommitFiberRoot == "function")
        try {
          Ol.onCommitFiberRoot(
            Pu,
            n,
            void 0,
            (n.current.flags & 128) === 128
          );
        } catch {
        }
      if (c !== null) {
        n = O.T, s = F.p, F.p = 2, O.T = null;
        try {
          for (var r = l.onRecoverableError, y = 0; y < c.length; y++) {
            var p = c[y];
            r(p.value, {
              componentStack: p.stack
            });
          }
        } finally {
          O.T = n, F.p = s;
        }
      }
      (Nl & 3) !== 0 && Nd(), pa(l), s = l.pendingLanes, (u & 4194090) !== 0 && (s & 42) !== 0 ? l === fm ? Lo++ : (Lo = 0, fm = l) : Lo = 0, Es(0);
    }
  }
  function vv(l, n) {
    (l.pooledCacheLanes &= n) === 0 && (n = l.pooledCache, n != null && (l.pooledCache = null, Yn(n)));
  }
  function Nd(l) {
    return pv(), Ss(), Hd(), pm();
  }
  function pm() {
    if (Hl !== 5) return !1;
    var l = tu, n = _d;
    _d = 0;
    var u = cn(Nl), c = O.T, s = F.p;
    try {
      F.p = 32 > u ? 32 : u, O.T = null, u = Ud, Ud = null;
      var r = tu, y = Nl;
      if (Hl = 0, Go = tu = null, Nl = 0, (gt & 6) !== 0) throw Error(_(331));
      var p = gt;
      if (gt |= 4, lm(r.current), em(
        r,
        r.current,
        y,
        u
      ), gt = p, Es(0, !1), Ol && typeof Ol.onPostCommitFiberRoot == "function")
        try {
          Ol.onPostCommitFiberRoot(Pu, r);
        } catch {
        }
      return !0;
    } finally {
      F.p = s, O.T = c, vv(l, n);
    }
  }
  function vm(l, n, u) {
    n = Oa(u, n), n = Yy(l.stateNode, n, 2), l = Vn(l, n, 2), l !== null && (Ne(l, 2), pa(l));
  }
  function Tt(l, n, u) {
    if (l.tag === 3)
      vm(l, l, u);
    else
      for (; n !== null; ) {
        if (n.tag === 3) {
          vm(
            n,
            l,
            u
          );
          break;
        } else if (n.tag === 1) {
          var c = n.stateNode;
          if (typeof n.type.getDerivedStateFromError == "function" || typeof c.componentDidCatch == "function" && (Di === null || !Di.has(c))) {
            l = Oa(u, l), u = Gy(2), c = Vn(n, u, 2), c !== null && (ha(
              u,
              c,
              n,
              l
            ), Ne(c, 2), pa(c));
            break;
          }
        }
        n = n.return;
      }
  }
  function wd(l, n, u) {
    var c = l.pingCache;
    if (c === null) {
      c = l.pingCache = new rv();
      var s = /* @__PURE__ */ new Set();
      c.set(n, s);
    } else
      s = c.get(n), s === void 0 && (s = /* @__PURE__ */ new Set(), c.set(n, s));
    s.has(u) || (um = !0, s.add(u), l = gm.bind(null, l, n, u), n.then(l, l));
  }
  function gm(l, n, u) {
    var c = l.pingCache;
    c !== null && c.delete(n), l.pingedLanes |= l.suspendedLanes & u, l.warmLanes &= ~u, _t === l && (lt & u) === u && ($t === 4 || $t === 3 && (lt & 62914560) === lt && 300 > pl() - cm ? (gt & 2) === 0 && lu(l, 0) : eu |= u, jo === lt && (jo = 0)), pa(l);
  }
  function bm(l, n) {
    n === 0 && (n = ue()), l = qn(l, n), l !== null && (Ne(l, n), pa(l));
  }
  function Cg(l) {
    var n = l.memoizedState, u = 0;
    n !== null && (u = n.retryLane), bm(l, u);
  }
  function xg(l, n) {
    var u = 0;
    switch (l.tag) {
      case 13:
        var c = l.stateNode, s = l.memoizedState;
        s !== null && (u = s.retryLane);
        break;
      case 19:
        c = l.stateNode;
        break;
      case 22:
        c = l.stateNode._retryCache;
        break;
      default:
        throw Error(_(314));
    }
    c !== null && c.delete(n), bm(l, u);
  }
  function Hg(l, n) {
    return zn(l, n);
  }
  var qd = null, Mi = null, Ts = !1, Qo = !1, jd = !1, _i = 0;
  function pa(l) {
    l !== Mi && l.next === null && (Mi === null ? qd = Mi = l : Mi = Mi.next = l), Qo = !0, Ts || (Ts = !0, Sv());
  }
  function Es(l, n) {
    if (!jd && Qo) {
      jd = !0;
      do
        for (var u = !1, c = qd; c !== null; ) {
          if (l !== 0) {
            var s = c.pendingLanes;
            if (s === 0) var r = 0;
            else {
              var y = c.suspendedLanes, p = c.pingedLanes;
              r = (1 << 31 - Dl(42 | l) + 1) - 1, r &= s & ~(y & ~p), r = r & 201326741 ? r & 201326741 | 1 : r ? r | 2 : 0;
            }
            r !== 0 && (u = !0, Rs(c, r));
          } else
            r = lt, r = un(
              c,
              c === _t ? r : 0,
              c.cancelPendingCommit !== null || c.timeoutHandle !== -1
            ), (r & 3) === 0 || m(c, r) || (u = !0, Rs(c, r));
          c = c.next;
        }
      while (u);
      jd = !1;
    }
  }
  function gv() {
    As();
  }
  function As() {
    Qo = Ts = !1;
    var l = 0;
    _i !== 0 && (Lu() && (l = _i), _i = 0);
    for (var n = pl(), u = null, c = qd; c !== null; ) {
      var s = c.next, r = Sm(c, n);
      r === 0 ? (c.next = null, u === null ? qd = s : u.next = s, s === null && (Mi = u)) : (u = c, (l !== 0 || (r & 3) !== 0) && (Qo = !0)), c = s;
    }
    Es(l);
  }
  function Sm(l, n) {
    for (var u = l.suspendedLanes, c = l.pingedLanes, s = l.expirationTimes, r = l.pendingLanes & -62914561; 0 < r; ) {
      var y = 31 - Dl(r), p = 1 << y, S = s[y];
      S === -1 ? ((p & u) === 0 || (p & c) !== 0) && (s[y] = D(p, n)) : S <= n && (l.expiredLanes |= p), r &= ~p;
    }
    if (n = _t, u = lt, u = un(
      l,
      l === n ? u : 0,
      l.cancelPendingCommit !== null || l.timeoutHandle !== -1
    ), c = l.callbackNode, u === 0 || l === n && (St === 2 || St === 9) || l.cancelPendingCommit !== null)
      return c !== null && c !== null && Pc(c), l.callbackNode = null, l.callbackPriority = 0;
    if ((u & 3) === 0 || m(l, u)) {
      if (n = u & -u, n === l.callbackPriority) return n;
      switch (c !== null && Pc(c), cn(u)) {
        case 2:
        case 8:
          u = Ke;
          break;
        case 32:
          u = Mn;
          break;
        case 268435456:
          u = gu;
          break;
        default:
          u = Mn;
      }
      return c = bv.bind(null, l), u = zn(u, c), l.callbackPriority = n, l.callbackNode = u, n;
    }
    return c !== null && c !== null && Pc(c), l.callbackPriority = 2, l.callbackNode = null, 2;
  }
  function bv(l, n) {
    if (Hl !== 0 && Hl !== 5)
      return l.callbackNode = null, l.callbackPriority = 0, null;
    var u = l.callbackNode;
    if (Nd() && l.callbackNode !== u)
      return null;
    var c = lt;
    return c = un(
      l,
      l === _t ? c : 0,
      l.cancelPendingCommit !== null || l.timeoutHandle !== -1
    ), c === 0 ? null : (Vo(l, c, n), Sm(l, pl()), l.callbackNode != null && l.callbackNode === u ? bv.bind(null, l) : null);
  }
  function Rs(l, n) {
    if (Nd()) return null;
    Vo(l, n, !0);
  }
  function Sv() {
    jg(function() {
      (gt & 6) !== 0 ? zn(
        ir,
        gv
      ) : As();
    });
  }
  function Hc() {
    return _i === 0 && (_i = le()), _i;
  }
  function Bd(l) {
    return l == null || typeof l == "symbol" || typeof l == "boolean" ? null : typeof l == "function" ? l : _f("" + l);
  }
  function Os(l, n) {
    var u = n.ownerDocument.createElement("input");
    return u.name = n.name, u.value = n.value, l.id && u.setAttribute("form", l.id), n.parentNode.insertBefore(u, n), l = new FormData(l), u.parentNode.removeChild(u), l;
  }
  function Tv(l, n, u, c, s) {
    if (n === "submit" && u && u.stateNode === s) {
      var r = Bd(
        (s[kl] || null).action
      ), y = c.submitter;
      y && (n = (n = y[kl] || null) ? Bd(n.formAction) : y.getAttribute("formAction"), n !== null && (r = n, y = null));
      var p = new Sr(
        "action",
        "action",
        null,
        c,
        s
      );
      l.push({
        event: p,
        listeners: [
          {
            instance: null,
            listener: function() {
              if (c.defaultPrevented) {
                if (_i !== 0) {
                  var S = y ? Os(s, y) : new FormData(s);
                  id(
                    u,
                    {
                      pending: !0,
                      data: S,
                      method: s.method,
                      action: r
                    },
                    null,
                    S
                  );
                }
              } else
                typeof r == "function" && (p.preventDefault(), S = y ? Os(s, y) : new FormData(s), id(
                  u,
                  {
                    pending: !0,
                    data: S,
                    method: s.method,
                    action: r
                  },
                  r,
                  S
                ));
            },
            currentTarget: s
          }
        ]
      });
    }
  }
  for (var Wt = 0; Wt < ro.length; Wt++) {
    var Ds = ro[Wt], Ng = Ds.toLowerCase(), Je = Ds[0].toUpperCase() + Ds.slice(1);
    La(
      Ng,
      "on" + Je
    );
  }
  La(Qp, "onAnimationEnd"), La(uy, "onAnimationIteration"), La(Zp, "onAnimationStart"), La("dblclick", "onDoubleClick"), La("focusin", "onFocus"), La("focusout", "onBlur"), La(iy, "onTransitionRun"), La(_r, "onTransitionStart"), La(Kp, "onTransitionCancel"), La(cy, "onTransitionEnd"), ti("onMouseEnter", ["mouseout", "mouseover"]), ti("onMouseLeave", ["mouseout", "mouseover"]), ti("onPointerEnter", ["pointerout", "pointerover"]), ti("onPointerLeave", ["pointerout", "pointerover"]), ei(
    "onChange",
    "change click focusin focusout input keydown keyup selectionchange".split(" ")
  ), ei(
    "onSelect",
    "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(
      " "
    )
  ), ei("onBeforeInput", [
    "compositionend",
    "keypress",
    "textInput",
    "paste"
  ]), ei(
    "onCompositionEnd",
    "compositionend focusout keydown keypress keyup mousedown".split(" ")
  ), ei(
    "onCompositionStart",
    "compositionstart focusout keydown keypress keyup mousedown".split(" ")
  ), ei(
    "onCompositionUpdate",
    "compositionupdate focusout keydown keypress keyup mousedown".split(" ")
  );
  var zs = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(
    " "
  ), Ui = new Set(
    "beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(zs)
  );
  function Nc(l, n) {
    n = (n & 4) !== 0;
    for (var u = 0; u < l.length; u++) {
      var c = l[u], s = c.event;
      c = c.listeners;
      e: {
        var r = void 0;
        if (n)
          for (var y = c.length - 1; 0 <= y; y--) {
            var p = c[y], S = p.instance, H = p.currentTarget;
            if (p = p.listener, S !== r && s.isPropagationStopped())
              break e;
            r = p, s.currentTarget = H;
            try {
              r(s);
            } catch (K) {
              ss(K);
            }
            s.currentTarget = null, r = S;
          }
        else
          for (y = 0; y < c.length; y++) {
            if (p = c[y], S = p.instance, H = p.currentTarget, p = p.listener, S !== r && s.isPropagationStopped())
              break e;
            r = p, s.currentTarget = H;
            try {
              r(s);
            } catch (K) {
              ss(K);
            }
            s.currentTarget = null, r = S;
          }
      }
    }
  }
  function Le(l, n) {
    var u = n[fr];
    u === void 0 && (u = n[fr] = /* @__PURE__ */ new Set());
    var c = l + "__bubble";
    u.has(c) || (Yd(n, l, 2, !1), u.add(c));
  }
  function Zo(l, n, u) {
    var c = 0;
    n && (c |= 4), Yd(
      u,
      l,
      c,
      n
    );
  }
  var Ko = "_reactListening" + Math.random().toString(36).slice(2);
  function Tm(l) {
    if (!l[Ko]) {
      l[Ko] = !0, Rf.forEach(function(u) {
        u !== "selectionchange" && (Ui.has(u) || Zo(u, !1, l), Zo(u, !0, l));
      });
      var n = l.nodeType === 9 ? l : l.ownerDocument;
      n === null || n[Ko] || (n[Ko] = !0, Zo("selectionchange", !1, n));
    }
  }
  function Yd(l, n, u, c) {
    switch (qm(n)) {
      case 2:
        var s = Uv;
        break;
      case 8:
        s = Cv;
        break;
      default:
        s = Nm;
    }
    u = s.bind(
      null,
      n,
      u,
      l
    ), s = void 0, !vr || n !== "touchstart" && n !== "touchmove" && n !== "wheel" || (s = !0), c ? s !== void 0 ? l.addEventListener(n, u, {
      capture: !0,
      passive: s
    }) : l.addEventListener(n, u, !0) : s !== void 0 ? l.addEventListener(n, u, {
      passive: s
    }) : l.addEventListener(n, u, !1);
  }
  function Ia(l, n, u, c, s) {
    var r = c;
    if ((n & 1) === 0 && (n & 2) === 0 && c !== null)
      e: for (; ; ) {
        if (c === null) return;
        var y = c.tag;
        if (y === 3 || y === 4) {
          var p = c.stateNode.containerInfo;
          if (p === s) break;
          if (y === 4)
            for (y = c.return; y !== null; ) {
              var S = y.tag;
              if ((S === 3 || S === 4) && y.stateNode.containerInfo === s)
                return;
              y = y.return;
            }
          for (; p !== null; ) {
            if (y = Ml(p), y === null) return;
            if (S = y.tag, S === 5 || S === 6 || S === 26 || S === 27) {
              c = r = y;
              continue e;
            }
            p = p.parentNode;
          }
        }
        c = c.return;
      }
    oo(function() {
      var H = r, K = pr(u), $ = [];
      e: {
        var q = oy.get(l);
        if (q !== void 0) {
          var Y = Sr, Ae = l;
          switch (l) {
            case "keypress":
              if (_l(u) === 0) break e;
            case "keydown":
            case "keyup":
              Y = on;
              break;
            case "focusin":
              Ae = "focus", Y = Xh;
              break;
            case "focusout":
              Ae = "blur", Y = Xh;
              break;
            case "beforeblur":
            case "afterblur":
              Y = Xh;
              break;
            case "click":
              if (u.button === 2) break e;
            case "auxclick":
            case "dblclick":
            case "mousedown":
            case "mousemove":
            case "mouseup":
            case "mouseout":
            case "mouseover":
            case "contextmenu":
              Y = Vh;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              Y = wp;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              Y = Kh;
              break;
            case Qp:
            case uy:
            case Zp:
              Y = Dg;
              break;
            case cy:
              Y = Gp;
              break;
            case "scroll":
            case "scrollend":
              Y = Hp;
              break;
            case "wheel":
              Y = ac;
              break;
            case "copy":
            case "cut":
            case "paste":
              Y = xf;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              Y = Hf;
              break;
            case "toggle":
            case "beforetoggle":
              Y = Lp;
          }
          var Re = (n & 4) !== 0, yt = !Re && (l === "scroll" || l === "scrollend"), M = Re ? q !== null ? q + "Capture" : null : q;
          Re = [];
          for (var R = H, C; R !== null; ) {
            var k = R;
            if (C = k.stateNode, k = k.tag, k !== 5 && k !== 26 && k !== 27 || C === null || M === null || (k = Pi(R, M), k != null && Re.push(
              Yu(R, k, C)
            )), yt) break;
            R = R.return;
          }
          0 < Re.length && (q = new Y(
            q,
            Ae,
            null,
            u,
            K
          ), $.push({ event: q, listeners: Re }));
        }
      }
      if ((n & 7) === 0) {
        e: {
          if (q = l === "mouseover" || l === "pointerover", Y = l === "mouseout" || l === "pointerout", q && u !== Ii && (Ae = u.relatedTarget || u.fromElement) && (Ml(Ae) || Ae[ao]))
            break e;
          if ((Y || q) && (q = K.window === K ? K : (q = K.ownerDocument) ? q.defaultView || q.parentWindow : window, Y ? (Ae = u.relatedTarget || u.toElement, Y = H, Ae = Ae ? Ml(Ae) : null, Ae !== null && (yt = Ee(Ae), Re = Ae.tag, Ae !== yt || Re !== 5 && Re !== 27 && Re !== 6) && (Ae = null)) : (Y = null, Ae = H), Y !== Ae)) {
            if (Re = Vh, k = "onMouseLeave", M = "onMouseEnter", R = "mouse", (l === "pointerout" || l === "pointerover") && (Re = Hf, k = "onPointerLeave", M = "onPointerEnter", R = "pointer"), yt = Y == null ? q : Af(Y), C = Ae == null ? q : Af(Ae), q = new Re(
              k,
              R + "leave",
              Y,
              u,
              K
            ), q.target = yt, q.relatedTarget = C, k = null, Ml(K) === H && (Re = new Re(
              M,
              R + "enter",
              Ae,
              u,
              K
            ), Re.target = C, Re.relatedTarget = yt, k = Re), yt = k, Y && Ae)
              t: {
                for (Re = Y, M = Ae, R = 0, C = Re; C; C = Ci(C))
                  R++;
                for (C = 0, k = M; k; k = Ci(k))
                  C++;
                for (; 0 < R - C; )
                  Re = Ci(Re), R--;
                for (; 0 < C - R; )
                  M = Ci(M), C--;
                for (; R--; ) {
                  if (Re === M || M !== null && Re === M.alternate)
                    break t;
                  Re = Ci(Re), M = Ci(M);
                }
                Re = null;
              }
            else Re = null;
            Y !== null && Ms(
              $,
              q,
              Y,
              Re,
              !1
            ), Ae !== null && yt !== null && Ms(
              $,
              yt,
              Ae,
              Re,
              !0
            );
          }
        }
        e: {
          if (q = H ? Af(H) : window, Y = q.nodeName && q.nodeName.toLowerCase(), Y === "select" || Y === "input" && q.type === "file")
            var de = Ih;
          else if (Or(q))
            if (Ph)
              de = ly;
            else {
              de = ci;
              var We = zr;
            }
          else
            Y = q.nodeName, !Y || Y.toLowerCase() !== "input" || q.type !== "checkbox" && q.type !== "radio" ? H && Fi(H.elementType) && (de = Ih) : de = Ru;
          if (de && (de = de(l, H))) {
            Dr(
              $,
              de,
              u,
              K
            );
            break e;
          }
          We && We(l, q, H), l === "focusout" && H && q.type === "number" && H.memoizedProps.value != null && zf(q, "number", q.value);
        }
        switch (We = H ? Af(H) : window, l) {
          case "focusin":
            (Or(We) || We.contentEditable === "true") && (Hn = We, rn = H, si = null);
            break;
          case "focusout":
            si = rn = Hn = null;
            break;
          case "mousedown":
            oc = !0;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            oc = !1, Mr($, u, K);
            break;
          case "selectionchange":
            if (cc) break;
          case "keydown":
          case "keyup":
            Mr($, u, K);
        }
        var Te;
        if (Nf)
          e: {
            switch (l) {
              case "compositionstart":
                var _e = "onCompositionStart";
                break e;
              case "compositionend":
                _e = "onCompositionEnd";
                break e;
              case "compositionupdate":
                _e = "onCompositionUpdate";
                break e;
            }
            _e = void 0;
          }
        else
          ii ? qf(l, u) && (_e = "onCompositionEnd") : l === "keydown" && u.keyCode === 229 && (_e = "onCompositionStart");
        _e && (Cn && u.locale !== "ko" && (ii || _e !== "onCompositionStart" ? _e === "onCompositionEnd" && ii && (Te = Gh()) : (Eu = K, fo = "value" in Eu ? Eu.value : Eu.textContent, ii = !0)), We = Jo(H, _e), 0 < We.length && (_e = new Qh(
          _e,
          l,
          null,
          u,
          K
        ), $.push({ event: _e, listeners: We }), Te ? _e.data = Te : (Te = ui(u), Te !== null && (_e.data = Te)))), (Te = kh ? Wh(l, u) : nc(l, u)) && (_e = Jo(H, "onBeforeInput"), 0 < _e.length && (We = new Qh(
          "onBeforeInput",
          "beforeinput",
          null,
          u,
          K
        ), $.push({
          event: We,
          listeners: _e
        }), We.data = Te)), Tv(
          $,
          l,
          H,
          u,
          K
        );
      }
      Nc($, n);
    });
  }
  function Yu(l, n, u) {
    return {
      instance: l,
      listener: n,
      currentTarget: u
    };
  }
  function Jo(l, n) {
    for (var u = n + "Capture", c = []; l !== null; ) {
      var s = l, r = s.stateNode;
      if (s = s.tag, s !== 5 && s !== 26 && s !== 27 || r === null || (s = Pi(l, u), s != null && c.unshift(
        Yu(l, s, r)
      ), s = Pi(l, n), s != null && c.push(
        Yu(l, s, r)
      )), l.tag === 3) return c;
      l = l.return;
    }
    return [];
  }
  function Ci(l) {
    if (l === null) return null;
    do
      l = l.return;
    while (l && l.tag !== 5 && l.tag !== 27);
    return l || null;
  }
  function Ms(l, n, u, c, s) {
    for (var r = n._reactName, y = []; u !== null && u !== c; ) {
      var p = u, S = p.alternate, H = p.stateNode;
      if (p = p.tag, S !== null && S === c) break;
      p !== 5 && p !== 26 && p !== 27 || H === null || (S = H, s ? (H = Pi(u, r), H != null && y.unshift(
        Yu(u, H, S)
      )) : s || (H = Pi(u, r), H != null && y.push(
        Yu(u, H, S)
      ))), u = u.return;
    }
    y.length !== 0 && l.push({ event: n, listeners: y });
  }
  var Ca = /\r\n?/g, Em = /\u0000|\uFFFD/g;
  function Ev(l) {
    return (typeof l == "string" ? l : "" + l).replace(Ca, `
`).replace(Em, "");
  }
  function Am(l, n) {
    return n = Ev(n), Ev(l) === n;
  }
  function Gd() {
  }
  function we(l, n, u, c, s, r) {
    switch (u) {
      case "children":
        typeof c == "string" ? n === "body" || n === "textarea" && c === "" || uo(l, c) : (typeof c == "number" || typeof c == "bigint") && n !== "body" && uo(l, "" + c);
        break;
      case "className":
        Of(l, "class", c);
        break;
      case "tabIndex":
        Of(l, "tabindex", c);
        break;
      case "dir":
      case "role":
      case "viewBox":
      case "width":
      case "height":
        Of(l, u, c);
        break;
      case "style":
        Mf(l, c, r);
        break;
      case "data":
        if (n !== "object") {
          Of(l, "data", c);
          break;
        }
      case "src":
      case "href":
        if (c === "" && (n !== "a" || u !== "href")) {
          l.removeAttribute(u);
          break;
        }
        if (c == null || typeof c == "function" || typeof c == "symbol" || typeof c == "boolean") {
          l.removeAttribute(u);
          break;
        }
        c = _f("" + c), l.setAttribute(u, c);
        break;
      case "action":
      case "formAction":
        if (typeof c == "function") {
          l.setAttribute(
            u,
            "javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')"
          );
          break;
        } else
          typeof r == "function" && (u === "formAction" ? (n !== "input" && we(l, n, "name", s.name, s, null), we(
            l,
            n,
            "formEncType",
            s.formEncType,
            s,
            null
          ), we(
            l,
            n,
            "formMethod",
            s.formMethod,
            s,
            null
          ), we(
            l,
            n,
            "formTarget",
            s.formTarget,
            s,
            null
          )) : (we(l, n, "encType", s.encType, s, null), we(l, n, "method", s.method, s, null), we(l, n, "target", s.target, s, null)));
        if (c == null || typeof c == "symbol" || typeof c == "boolean") {
          l.removeAttribute(u);
          break;
        }
        c = _f("" + c), l.setAttribute(u, c);
        break;
      case "onClick":
        c != null && (l.onclick = Gd);
        break;
      case "onScroll":
        c != null && Le("scroll", l);
        break;
      case "onScrollEnd":
        c != null && Le("scrollend", l);
        break;
      case "dangerouslySetInnerHTML":
        if (c != null) {
          if (typeof c != "object" || !("__html" in c))
            throw Error(_(61));
          if (u = c.__html, u != null) {
            if (s.children != null) throw Error(_(60));
            l.innerHTML = u;
          }
        }
        break;
      case "multiple":
        l.multiple = c && typeof c != "function" && typeof c != "symbol";
        break;
      case "muted":
        l.muted = c && typeof c != "function" && typeof c != "symbol";
        break;
      case "suppressContentEditableWarning":
      case "suppressHydrationWarning":
      case "defaultValue":
      case "defaultChecked":
      case "innerHTML":
      case "ref":
        break;
      case "autoFocus":
        break;
      case "xlinkHref":
        if (c == null || typeof c == "function" || typeof c == "boolean" || typeof c == "symbol") {
          l.removeAttribute("xlink:href");
          break;
        }
        u = _f("" + c), l.setAttributeNS(
          "http://www.w3.org/1999/xlink",
          "xlink:href",
          u
        );
        break;
      case "contentEditable":
      case "spellCheck":
      case "draggable":
      case "value":
      case "autoReverse":
      case "externalResourcesRequired":
      case "focusable":
      case "preserveAlpha":
        c != null && typeof c != "function" && typeof c != "symbol" ? l.setAttribute(u, "" + c) : l.removeAttribute(u);
        break;
      case "inert":
      case "allowFullScreen":
      case "async":
      case "autoPlay":
      case "controls":
      case "default":
      case "defer":
      case "disabled":
      case "disablePictureInPicture":
      case "disableRemotePlayback":
      case "formNoValidate":
      case "hidden":
      case "loop":
      case "noModule":
      case "noValidate":
      case "open":
      case "playsInline":
      case "readOnly":
      case "required":
      case "reversed":
      case "scoped":
      case "seamless":
      case "itemScope":
        c && typeof c != "function" && typeof c != "symbol" ? l.setAttribute(u, "") : l.removeAttribute(u);
        break;
      case "capture":
      case "download":
        c === !0 ? l.setAttribute(u, "") : c !== !1 && c != null && typeof c != "function" && typeof c != "symbol" ? l.setAttribute(u, c) : l.removeAttribute(u);
        break;
      case "cols":
      case "rows":
      case "size":
      case "span":
        c != null && typeof c != "function" && typeof c != "symbol" && !isNaN(c) && 1 <= c ? l.setAttribute(u, c) : l.removeAttribute(u);
        break;
      case "rowSpan":
      case "start":
        c == null || typeof c == "function" || typeof c == "symbol" || isNaN(c) ? l.removeAttribute(u) : l.setAttribute(u, c);
        break;
      case "popover":
        Le("beforetoggle", l), Le("toggle", l), Su(l, "popover", c);
        break;
      case "xlinkActuate":
        _n(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:actuate",
          c
        );
        break;
      case "xlinkArcrole":
        _n(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:arcrole",
          c
        );
        break;
      case "xlinkRole":
        _n(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:role",
          c
        );
        break;
      case "xlinkShow":
        _n(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:show",
          c
        );
        break;
      case "xlinkTitle":
        _n(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:title",
          c
        );
        break;
      case "xlinkType":
        _n(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:type",
          c
        );
        break;
      case "xmlBase":
        _n(
          l,
          "http://www.w3.org/XML/1998/namespace",
          "xml:base",
          c
        );
        break;
      case "xmlLang":
        _n(
          l,
          "http://www.w3.org/XML/1998/namespace",
          "xml:lang",
          c
        );
        break;
      case "xmlSpace":
        _n(
          l,
          "http://www.w3.org/XML/1998/namespace",
          "xml:space",
          c
        );
        break;
      case "is":
        Su(l, "is", c);
        break;
      case "innerText":
      case "textContent":
        break;
      default:
        (!(2 < u.length) || u[0] !== "o" && u[0] !== "O" || u[1] !== "n" && u[1] !== "N") && (u = Ag.get(u) || u, Su(l, u, c));
    }
  }
  function L(l, n, u, c, s, r) {
    switch (u) {
      case "style":
        Mf(l, c, r);
        break;
      case "dangerouslySetInnerHTML":
        if (c != null) {
          if (typeof c != "object" || !("__html" in c))
            throw Error(_(61));
          if (u = c.__html, u != null) {
            if (s.children != null) throw Error(_(60));
            l.innerHTML = u;
          }
        }
        break;
      case "children":
        typeof c == "string" ? uo(l, c) : (typeof c == "number" || typeof c == "bigint") && uo(l, "" + c);
        break;
      case "onScroll":
        c != null && Le("scroll", l);
        break;
      case "onScrollEnd":
        c != null && Le("scrollend", l);
        break;
      case "onClick":
        c != null && (l.onclick = Gd);
        break;
      case "suppressContentEditableWarning":
      case "suppressHydrationWarning":
      case "innerHTML":
      case "ref":
        break;
      case "innerText":
      case "textContent":
        break;
      default:
        if (!Aa.hasOwnProperty(u))
          e: {
            if (u[0] === "o" && u[1] === "n" && (s = u.endsWith("Capture"), n = u.slice(2, s ? u.length - 7 : void 0), r = l[kl] || null, r = r != null ? r[u] : null, typeof r == "function" && l.removeEventListener(n, r, s), typeof c == "function")) {
              typeof r != "function" && r !== null && (u in l ? l[u] = null : l.hasAttribute(u) && l.removeAttribute(u)), l.addEventListener(n, c, s);
              break e;
            }
            u in l ? l[u] = c : c === !0 ? l.setAttribute(u, "") : Su(l, u, c);
          }
    }
  }
  function Ce(l, n, u) {
    switch (n) {
      case "div":
      case "span":
      case "svg":
      case "path":
      case "a":
      case "g":
      case "p":
      case "li":
        break;
      case "img":
        Le("error", l), Le("load", l);
        var c = !1, s = !1, r;
        for (r in u)
          if (u.hasOwnProperty(r)) {
            var y = u[r];
            if (y != null)
              switch (r) {
                case "src":
                  c = !0;
                  break;
                case "srcSet":
                  s = !0;
                  break;
                case "children":
                case "dangerouslySetInnerHTML":
                  throw Error(_(137, n));
                default:
                  we(l, n, r, y, u, null);
              }
          }
        s && we(l, n, "srcSet", u.srcSet, u, null), c && we(l, n, "src", u.src, u, null);
        return;
      case "input":
        Le("invalid", l);
        var p = r = y = s = null, S = null, H = null;
        for (c in u)
          if (u.hasOwnProperty(c)) {
            var K = u[c];
            if (K != null)
              switch (c) {
                case "name":
                  s = K;
                  break;
                case "type":
                  y = K;
                  break;
                case "checked":
                  S = K;
                  break;
                case "defaultChecked":
                  H = K;
                  break;
                case "value":
                  r = K;
                  break;
                case "defaultValue":
                  p = K;
                  break;
                case "children":
                case "dangerouslySetInnerHTML":
                  if (K != null)
                    throw Error(_(137, n));
                  break;
                default:
                  we(l, n, c, K, u, null);
              }
          }
        yr(
          l,
          r,
          p,
          S,
          H,
          y,
          s,
          !1
        ), ai(l);
        return;
      case "select":
        Le("invalid", l), c = y = r = null;
        for (s in u)
          if (u.hasOwnProperty(s) && (p = u[s], p != null))
            switch (s) {
              case "value":
                r = p;
                break;
              case "defaultValue":
                y = p;
                break;
              case "multiple":
                c = p;
              default:
                we(l, n, s, p, u, null);
            }
        n = r, u = y, l.multiple = !!c, n != null ? Wi(l, !!c, n, !1) : u != null && Wi(l, !!c, u, !0);
        return;
      case "textarea":
        Le("invalid", l), r = s = c = null;
        for (y in u)
          if (u.hasOwnProperty(y) && (p = u[y], p != null))
            switch (y) {
              case "value":
                c = p;
                break;
              case "defaultValue":
                s = p;
                break;
              case "children":
                r = p;
                break;
              case "dangerouslySetInnerHTML":
                if (p != null) throw Error(_(91));
                break;
              default:
                we(l, n, y, p, u, null);
            }
        Bh(l, c, s, r), ai(l);
        return;
      case "option":
        for (S in u)
          if (u.hasOwnProperty(S) && (c = u[S], c != null))
            switch (S) {
              case "selected":
                l.selected = c && typeof c != "function" && typeof c != "symbol";
                break;
              default:
                we(l, n, S, c, u, null);
            }
        return;
      case "dialog":
        Le("beforetoggle", l), Le("toggle", l), Le("cancel", l), Le("close", l);
        break;
      case "iframe":
      case "object":
        Le("load", l);
        break;
      case "video":
      case "audio":
        for (c = 0; c < zs.length; c++)
          Le(zs[c], l);
        break;
      case "image":
        Le("error", l), Le("load", l);
        break;
      case "details":
        Le("toggle", l);
        break;
      case "embed":
      case "source":
      case "link":
        Le("error", l), Le("load", l);
      case "area":
      case "base":
      case "br":
      case "col":
      case "hr":
      case "keygen":
      case "meta":
      case "param":
      case "track":
      case "wbr":
      case "menuitem":
        for (H in u)
          if (u.hasOwnProperty(H) && (c = u[H], c != null))
            switch (H) {
              case "children":
              case "dangerouslySetInnerHTML":
                throw Error(_(137, n));
              default:
                we(l, n, H, c, u, null);
            }
        return;
      default:
        if (Fi(n)) {
          for (K in u)
            u.hasOwnProperty(K) && (c = u[K], c !== void 0 && L(
              l,
              n,
              K,
              c,
              u,
              void 0
            ));
          return;
        }
    }
    for (p in u)
      u.hasOwnProperty(p) && (c = u[p], c != null && we(l, n, p, c, u, null));
  }
  function wg(l, n, u, c) {
    switch (n) {
      case "div":
      case "span":
      case "svg":
      case "path":
      case "a":
      case "g":
      case "p":
      case "li":
        break;
      case "input":
        var s = null, r = null, y = null, p = null, S = null, H = null, K = null;
        for (Y in u) {
          var $ = u[Y];
          if (u.hasOwnProperty(Y) && $ != null)
            switch (Y) {
              case "checked":
                break;
              case "value":
                break;
              case "defaultValue":
                S = $;
              default:
                c.hasOwnProperty(Y) || we(l, n, Y, null, c, $);
            }
        }
        for (var q in c) {
          var Y = c[q];
          if ($ = u[q], c.hasOwnProperty(q) && (Y != null || $ != null))
            switch (q) {
              case "type":
                r = Y;
                break;
              case "name":
                s = Y;
                break;
              case "checked":
                H = Y;
                break;
              case "defaultChecked":
                K = Y;
                break;
              case "value":
                y = Y;
                break;
              case "defaultValue":
                p = Y;
                break;
              case "children":
              case "dangerouslySetInnerHTML":
                if (Y != null)
                  throw Error(_(137, n));
                break;
              default:
                Y !== $ && we(
                  l,
                  n,
                  q,
                  Y,
                  c,
                  $
                );
            }
        }
        hr(
          l,
          y,
          p,
          S,
          H,
          K,
          r,
          s
        );
        return;
      case "select":
        Y = y = p = q = null;
        for (r in u)
          if (S = u[r], u.hasOwnProperty(r) && S != null)
            switch (r) {
              case "value":
                break;
              case "multiple":
                Y = S;
              default:
                c.hasOwnProperty(r) || we(
                  l,
                  n,
                  r,
                  null,
                  c,
                  S
                );
            }
        for (s in c)
          if (r = c[s], S = u[s], c.hasOwnProperty(s) && (r != null || S != null))
            switch (s) {
              case "value":
                q = r;
                break;
              case "defaultValue":
                p = r;
                break;
              case "multiple":
                y = r;
              default:
                r !== S && we(
                  l,
                  n,
                  s,
                  r,
                  c,
                  S
                );
            }
        n = p, u = y, c = Y, q != null ? Wi(l, !!u, q, !1) : !!c != !!u && (n != null ? Wi(l, !!u, n, !0) : Wi(l, !!u, u ? [] : "", !1));
        return;
      case "textarea":
        Y = q = null;
        for (p in u)
          if (s = u[p], u.hasOwnProperty(p) && s != null && !c.hasOwnProperty(p))
            switch (p) {
              case "value":
                break;
              case "children":
                break;
              default:
                we(l, n, p, null, c, s);
            }
        for (y in c)
          if (s = c[y], r = u[y], c.hasOwnProperty(y) && (s != null || r != null))
            switch (y) {
              case "value":
                q = s;
                break;
              case "defaultValue":
                Y = s;
                break;
              case "children":
                break;
              case "dangerouslySetInnerHTML":
                if (s != null) throw Error(_(91));
                break;
              default:
                s !== r && we(l, n, y, s, c, r);
            }
        jh(l, q, Y);
        return;
      case "option":
        for (var Ae in u)
          if (q = u[Ae], u.hasOwnProperty(Ae) && q != null && !c.hasOwnProperty(Ae))
            switch (Ae) {
              case "selected":
                l.selected = !1;
                break;
              default:
                we(
                  l,
                  n,
                  Ae,
                  null,
                  c,
                  q
                );
            }
        for (S in c)
          if (q = c[S], Y = u[S], c.hasOwnProperty(S) && q !== Y && (q != null || Y != null))
            switch (S) {
              case "selected":
                l.selected = q && typeof q != "function" && typeof q != "symbol";
                break;
              default:
                we(
                  l,
                  n,
                  S,
                  q,
                  c,
                  Y
                );
            }
        return;
      case "img":
      case "link":
      case "area":
      case "base":
      case "br":
      case "col":
      case "embed":
      case "hr":
      case "keygen":
      case "meta":
      case "param":
      case "source":
      case "track":
      case "wbr":
      case "menuitem":
        for (var Re in u)
          q = u[Re], u.hasOwnProperty(Re) && q != null && !c.hasOwnProperty(Re) && we(l, n, Re, null, c, q);
        for (H in c)
          if (q = c[H], Y = u[H], c.hasOwnProperty(H) && q !== Y && (q != null || Y != null))
            switch (H) {
              case "children":
              case "dangerouslySetInnerHTML":
                if (q != null)
                  throw Error(_(137, n));
                break;
              default:
                we(
                  l,
                  n,
                  H,
                  q,
                  c,
                  Y
                );
            }
        return;
      default:
        if (Fi(n)) {
          for (var yt in u)
            q = u[yt], u.hasOwnProperty(yt) && q !== void 0 && !c.hasOwnProperty(yt) && L(
              l,
              n,
              yt,
              void 0,
              c,
              q
            );
          for (K in c)
            q = c[K], Y = u[K], !c.hasOwnProperty(K) || q === Y || q === void 0 && Y === void 0 || L(
              l,
              n,
              K,
              q,
              c,
              Y
            );
          return;
        }
    }
    for (var M in u)
      q = u[M], u.hasOwnProperty(M) && q != null && !c.hasOwnProperty(M) && we(l, n, M, null, c, q);
    for ($ in c)
      q = c[$], Y = u[$], !c.hasOwnProperty($) || q === Y || q == null && Y == null || we(l, n, $, q, c, Y);
  }
  var _s = null, Us = null;
  function Pa(l) {
    return l.nodeType === 9 ? l : l.ownerDocument;
  }
  function Gu(l) {
    switch (l) {
      case "http://www.w3.org/2000/svg":
        return 1;
      case "http://www.w3.org/1998/Math/MathML":
        return 2;
      default:
        return 0;
    }
  }
  function ko(l, n) {
    if (l === 0)
      switch (n) {
        case "svg":
          return 1;
        case "math":
          return 2;
        default:
          return 0;
      }
    return l === 1 && n === "foreignObject" ? 0 : l;
  }
  function au(l, n) {
    return l === "textarea" || l === "noscript" || typeof n.children == "string" || typeof n.children == "number" || typeof n.children == "bigint" || typeof n.dangerouslySetInnerHTML == "object" && n.dangerouslySetInnerHTML !== null && n.dangerouslySetInnerHTML.__html != null;
  }
  var $o = null;
  function Lu() {
    var l = window.event;
    return l && l.type === "popstate" ? l === $o ? !1 : ($o = l, !0) : ($o = null, !1);
  }
  var Ld = typeof setTimeout == "function" ? setTimeout : void 0, qg = typeof clearTimeout == "function" ? clearTimeout : void 0, Av = typeof Promise == "function" ? Promise : void 0, jg = typeof queueMicrotask == "function" ? queueMicrotask : typeof Av < "u" ? function(l) {
    return Av.resolve(null).then(l).catch(nu);
  } : Ld;
  function nu(l) {
    setTimeout(function() {
      throw l;
    });
  }
  function xi(l) {
    return l === "head";
  }
  function Vd(l, n) {
    var u = n, c = 0, s = 0;
    do {
      var r = u.nextSibling;
      if (l.removeChild(u), r && r.nodeType === 8)
        if (u = r.data, u === "/$") {
          if (0 < c && 8 > c) {
            u = c;
            var y = l.ownerDocument;
            if (u & 1 && va(y.documentElement), u & 2 && va(y.body), u & 4)
              for (u = y.head, va(u), y = u.firstChild; y; ) {
                var p = y.nextSibling, S = y.nodeName;
                y[ye] || S === "SCRIPT" || S === "STYLE" || S === "LINK" && y.rel.toLowerCase() === "stylesheet" || u.removeChild(y), y = p;
              }
          }
          if (s === 0) {
            l.removeChild(r), iu(n);
            return;
          }
          s--;
        } else
          u === "$" || u === "$?" || u === "$!" ? s++ : c = u.charCodeAt(0) - 48;
      else c = 0;
      u = r;
    } while (u);
    iu(n);
  }
  function Cs(l) {
    var n = l.firstChild;
    for (n && n.nodeType === 10 && (n = n.nextSibling); n; ) {
      var u = n;
      switch (n = n.nextSibling, u.nodeName) {
        case "HTML":
        case "HEAD":
        case "BODY":
          Cs(u), Ef(u);
          continue;
        case "SCRIPT":
        case "STYLE":
          continue;
        case "LINK":
          if (u.rel.toLowerCase() === "stylesheet") continue;
      }
      l.removeChild(u);
    }
  }
  function Wo(l, n, u, c) {
    for (; l.nodeType === 1; ) {
      var s = u;
      if (l.nodeName.toLowerCase() !== n.toLowerCase()) {
        if (!c && (l.nodeName !== "INPUT" || l.type !== "hidden"))
          break;
      } else if (c) {
        if (!l[ye])
          switch (n) {
            case "meta":
              if (!l.hasAttribute("itemprop")) break;
              return l;
            case "link":
              if (r = l.getAttribute("rel"), r === "stylesheet" && l.hasAttribute("data-precedence"))
                break;
              if (r !== s.rel || l.getAttribute("href") !== (s.href == null || s.href === "" ? null : s.href) || l.getAttribute("crossorigin") !== (s.crossOrigin == null ? null : s.crossOrigin) || l.getAttribute("title") !== (s.title == null ? null : s.title))
                break;
              return l;
            case "style":
              if (l.hasAttribute("data-precedence")) break;
              return l;
            case "script":
              if (r = l.getAttribute("src"), (r !== (s.src == null ? null : s.src) || l.getAttribute("type") !== (s.type == null ? null : s.type) || l.getAttribute("crossorigin") !== (s.crossOrigin == null ? null : s.crossOrigin)) && r && l.hasAttribute("async") && !l.hasAttribute("itemprop"))
                break;
              return l;
            default:
              return l;
          }
      } else if (n === "input" && l.type === "hidden") {
        var r = s.name == null ? null : "" + s.name;
        if (s.type === "hidden" && l.getAttribute("name") === r)
          return l;
      } else return l;
      if (l = Tn(l.nextSibling), l === null) break;
    }
    return null;
  }
  function Bg(l, n, u) {
    if (n === "") return null;
    for (; l.nodeType !== 3; )
      if ((l.nodeType !== 1 || l.nodeName !== "INPUT" || l.type !== "hidden") && !u || (l = Tn(l.nextSibling), l === null)) return null;
    return l;
  }
  function xs(l) {
    return l.data === "$!" || l.data === "$?" && l.ownerDocument.readyState === "complete";
  }
  function Yg(l, n) {
    var u = l.ownerDocument;
    if (l.data !== "$?" || u.readyState === "complete")
      n();
    else {
      var c = function() {
        n(), u.removeEventListener("DOMContentLoaded", c);
      };
      u.addEventListener("DOMContentLoaded", c), l._reactRetry = c;
    }
  }
  function Tn(l) {
    for (; l != null; l = l.nextSibling) {
      var n = l.nodeType;
      if (n === 1 || n === 3) break;
      if (n === 8) {
        if (n = l.data, n === "$" || n === "$!" || n === "$?" || n === "F!" || n === "F")
          break;
        if (n === "/$") return null;
      }
    }
    return l;
  }
  var Hi = null;
  function wl(l) {
    l = l.previousSibling;
    for (var n = 0; l; ) {
      if (l.nodeType === 8) {
        var u = l.data;
        if (u === "$" || u === "$!" || u === "$?") {
          if (n === 0) return l;
          n--;
        } else u === "/$" && n++;
      }
      l = l.previousSibling;
    }
    return null;
  }
  function fe(l, n, u) {
    switch (n = Pa(u), l) {
      case "html":
        if (l = n.documentElement, !l) throw Error(_(452));
        return l;
      case "head":
        if (l = n.head, !l) throw Error(_(453));
        return l;
      case "body":
        if (l = n.body, !l) throw Error(_(454));
        return l;
      default:
        throw Error(_(451));
    }
  }
  function va(l) {
    for (var n = l.attributes; n.length; )
      l.removeAttributeNode(n[0]);
    Ef(l);
  }
  var Ft = /* @__PURE__ */ new Map(), Ql = /* @__PURE__ */ new Set();
  function Xd(l) {
    return typeof l.getRootNode == "function" ? l.getRootNode() : l.nodeType === 9 ? l : l.ownerDocument;
  }
  var Vu = F.d;
  F.d = {
    f: Qd,
    r: Zd,
    D: Xu,
    C: Kd,
    L: Ni,
    m: Zl,
    X: wi,
    S: ga,
    M: Rm
  };
  function Qd() {
    var l = Vu.f(), n = Uc();
    return l || n;
  }
  function Zd(l) {
    var n = Ki(l);
    n !== null && n.tag === 5 && n.type === "form" ? Mo(n) : Vu.r(l);
  }
  var ql = typeof document > "u" ? null : document;
  function En(l, n, u) {
    var c = ql;
    if (c && typeof n == "string" && n) {
      var s = Ya(n);
      s = 'link[rel="' + l + '"][href="' + s + '"]', typeof u == "string" && (s += '[crossorigin="' + u + '"]'), Ql.has(s) || (Ql.add(s), l = { rel: l, crossOrigin: u, href: n }, c.querySelector(s) === null && (n = c.createElement("link"), Ce(n, "link", l), ol(n), c.head.appendChild(n)));
    }
  }
  function Xu(l) {
    Vu.D(l), En("dns-prefetch", l, null);
  }
  function Kd(l, n) {
    Vu.C(l, n), En("preconnect", l, n);
  }
  function Ni(l, n, u) {
    Vu.L(l, n, u);
    var c = ql;
    if (c && l && n) {
      var s = 'link[rel="preload"][as="' + Ya(n) + '"]';
      n === "image" && u && u.imageSrcSet ? (s += '[imagesrcset="' + Ya(
        u.imageSrcSet
      ) + '"]', typeof u.imageSizes == "string" && (s += '[imagesizes="' + Ya(
        u.imageSizes
      ) + '"]')) : s += '[href="' + Ya(l) + '"]';
      var r = s;
      switch (n) {
        case "style":
          r = Fo(l);
          break;
        case "script":
          r = en(l);
      }
      Ft.has(r) || (l = te(
        {
          rel: "preload",
          href: n === "image" && u && u.imageSrcSet ? void 0 : l,
          as: n
        },
        u
      ), Ft.set(r, l), c.querySelector(s) !== null || n === "style" && c.querySelector(Io(r)) || n === "script" && c.querySelector(wc(r)) || (n = c.createElement("link"), Ce(n, "link", l), ol(n), c.head.appendChild(n)));
    }
  }
  function Zl(l, n) {
    Vu.m(l, n);
    var u = ql;
    if (u && l) {
      var c = n && typeof n.as == "string" ? n.as : "script", s = 'link[rel="modulepreload"][as="' + Ya(c) + '"][href="' + Ya(l) + '"]', r = s;
      switch (c) {
        case "audioworklet":
        case "paintworklet":
        case "serviceworker":
        case "sharedworker":
        case "worker":
        case "script":
          r = en(l);
      }
      if (!Ft.has(r) && (l = te({ rel: "modulepreload", href: l }, n), Ft.set(r, l), u.querySelector(s) === null)) {
        switch (c) {
          case "audioworklet":
          case "paintworklet":
          case "serviceworker":
          case "sharedworker":
          case "worker":
          case "script":
            if (u.querySelector(wc(r)))
              return;
        }
        c = u.createElement("link"), Ce(c, "link", l), ol(c), u.head.appendChild(c);
      }
    }
  }
  function ga(l, n, u) {
    Vu.S(l, n, u);
    var c = ql;
    if (c && l) {
      var s = bu(c).hoistableStyles, r = Fo(l);
      n = n || "default";
      var y = s.get(r);
      if (!y) {
        var p = { loading: 0, preload: null };
        if (y = c.querySelector(
          Io(r)
        ))
          p.loading = 5;
        else {
          l = te(
            { rel: "stylesheet", href: l, "data-precedence": n },
            u
          ), (u = Ft.get(r)) && kd(l, u);
          var S = y = c.createElement("link");
          ol(S), Ce(S, "link", l), S._p = new Promise(function(H, K) {
            S.onload = H, S.onerror = K;
          }), S.addEventListener("load", function() {
            p.loading |= 1;
          }), S.addEventListener("error", function() {
            p.loading |= 2;
          }), p.loading |= 4, Jd(y, n, c);
        }
        y = {
          type: "stylesheet",
          instance: y,
          count: 1,
          state: p
        }, s.set(r, y);
      }
    }
  }
  function wi(l, n) {
    Vu.X(l, n);
    var u = ql;
    if (u && l) {
      var c = bu(u).hoistableScripts, s = en(l), r = c.get(s);
      r || (r = u.querySelector(wc(s)), r || (l = te({ src: l, async: !0 }, n), (n = Ft.get(s)) && $d(l, n), r = u.createElement("script"), ol(r), Ce(r, "link", l), u.head.appendChild(r)), r = {
        type: "script",
        instance: r,
        count: 1,
        state: null
      }, c.set(s, r));
    }
  }
  function Rm(l, n) {
    Vu.M(l, n);
    var u = ql;
    if (u && l) {
      var c = bu(u).hoistableScripts, s = en(l), r = c.get(s);
      r || (r = u.querySelector(wc(s)), r || (l = te({ src: l, async: !0, type: "module" }, n), (n = Ft.get(s)) && $d(l, n), r = u.createElement("script"), ol(r), Ce(r, "link", l), u.head.appendChild(r)), r = {
        type: "script",
        instance: r,
        count: 1,
        state: null
      }, c.set(s, r));
    }
  }
  function Rv(l, n, u, c) {
    var s = (s = oe.current) ? Xd(s) : null;
    if (!s) throw Error(_(446));
    switch (l) {
      case "meta":
      case "title":
        return null;
      case "style":
        return typeof u.precedence == "string" && typeof u.href == "string" ? (n = Fo(u.href), u = bu(
          s
        ).hoistableStyles, c = u.get(n), c || (c = {
          type: "style",
          instance: null,
          count: 0,
          state: null
        }, u.set(n, c)), c) : { type: "void", instance: null, count: 0, state: null };
      case "link":
        if (u.rel === "stylesheet" && typeof u.href == "string" && typeof u.precedence == "string") {
          l = Fo(u.href);
          var r = bu(
            s
          ).hoistableStyles, y = r.get(l);
          if (y || (s = s.ownerDocument || s, y = {
            type: "stylesheet",
            instance: null,
            count: 0,
            state: { loading: 0, preload: null }
          }, r.set(l, y), (r = s.querySelector(
            Io(l)
          )) && !r._p && (y.instance = r, y.state.loading = 5), Ft.has(l) || (u = {
            rel: "preload",
            as: "style",
            href: u.href,
            crossOrigin: u.crossOrigin,
            integrity: u.integrity,
            media: u.media,
            hrefLang: u.hrefLang,
            referrerPolicy: u.referrerPolicy
          }, Ft.set(l, u), r || Ov(
            s,
            l,
            u,
            y.state
          ))), n && c === null)
            throw Error(_(528, ""));
          return y;
        }
        if (n && c !== null)
          throw Error(_(529, ""));
        return null;
      case "script":
        return n = u.async, u = u.src, typeof u == "string" && n && typeof n != "function" && typeof n != "symbol" ? (n = en(u), u = bu(
          s
        ).hoistableScripts, c = u.get(n), c || (c = {
          type: "script",
          instance: null,
          count: 0,
          state: null
        }, u.set(n, c)), c) : { type: "void", instance: null, count: 0, state: null };
      default:
        throw Error(_(444, l));
    }
  }
  function Fo(l) {
    return 'href="' + Ya(l) + '"';
  }
  function Io(l) {
    return 'link[rel="stylesheet"][' + l + "]";
  }
  function Po(l) {
    return te({}, l, {
      "data-precedence": l.precedence,
      precedence: null
    });
  }
  function Ov(l, n, u, c) {
    l.querySelector('link[rel="preload"][as="style"][' + n + "]") ? c.loading = 1 : (n = l.createElement("link"), c.preload = n, n.addEventListener("load", function() {
      return c.loading |= 1;
    }), n.addEventListener("error", function() {
      return c.loading |= 2;
    }), Ce(n, "link", u), ol(n), l.head.appendChild(n));
  }
  function en(l) {
    return '[src="' + Ya(l) + '"]';
  }
  function wc(l) {
    return "script[async]" + l;
  }
  function Dv(l, n, u) {
    if (n.count++, n.instance === null)
      switch (n.type) {
        case "style":
          var c = l.querySelector(
            'style[data-href~="' + Ya(u.href) + '"]'
          );
          if (c)
            return n.instance = c, ol(c), c;
          var s = te({}, u, {
            "data-href": u.href,
            "data-precedence": u.precedence,
            href: null,
            precedence: null
          });
          return c = (l.ownerDocument || l).createElement(
            "style"
          ), ol(c), Ce(c, "style", s), Jd(c, u.precedence, l), n.instance = c;
        case "stylesheet":
          s = Fo(u.href);
          var r = l.querySelector(
            Io(s)
          );
          if (r)
            return n.state.loading |= 4, n.instance = r, ol(r), r;
          c = Po(u), (s = Ft.get(s)) && kd(c, s), r = (l.ownerDocument || l).createElement("link"), ol(r);
          var y = r;
          return y._p = new Promise(function(p, S) {
            y.onload = p, y.onerror = S;
          }), Ce(r, "link", c), n.state.loading |= 4, Jd(r, u.precedence, l), n.instance = r;
        case "script":
          return r = en(u.src), (s = l.querySelector(
            wc(r)
          )) ? (n.instance = s, ol(s), s) : (c = u, (s = Ft.get(r)) && (c = te({}, u), $d(c, s)), l = l.ownerDocument || l, s = l.createElement("script"), ol(s), Ce(s, "link", c), l.head.appendChild(s), n.instance = s);
        case "void":
          return null;
        default:
          throw Error(_(443, n.type));
      }
    else
      n.type === "stylesheet" && (n.state.loading & 4) === 0 && (c = n.instance, n.state.loading |= 4, Jd(c, u.precedence, l));
    return n.instance;
  }
  function Jd(l, n, u) {
    for (var c = u.querySelectorAll(
      'link[rel="stylesheet"][data-precedence],style[data-precedence]'
    ), s = c.length ? c[c.length - 1] : null, r = s, y = 0; y < c.length; y++) {
      var p = c[y];
      if (p.dataset.precedence === n) r = p;
      else if (r !== s) break;
    }
    r ? r.parentNode.insertBefore(l, r.nextSibling) : (n = u.nodeType === 9 ? u.head : u, n.insertBefore(l, n.firstChild));
  }
  function kd(l, n) {
    l.crossOrigin == null && (l.crossOrigin = n.crossOrigin), l.referrerPolicy == null && (l.referrerPolicy = n.referrerPolicy), l.title == null && (l.title = n.title);
  }
  function $d(l, n) {
    l.crossOrigin == null && (l.crossOrigin = n.crossOrigin), l.referrerPolicy == null && (l.referrerPolicy = n.referrerPolicy), l.integrity == null && (l.integrity = n.integrity);
  }
  var qi = null;
  function Om(l, n, u) {
    if (qi === null) {
      var c = /* @__PURE__ */ new Map(), s = qi = /* @__PURE__ */ new Map();
      s.set(u, c);
    } else
      s = qi, c = s.get(u), c || (c = /* @__PURE__ */ new Map(), s.set(u, c));
    if (c.has(l)) return c;
    for (c.set(l, null), u = u.getElementsByTagName(l), s = 0; s < u.length; s++) {
      var r = u[s];
      if (!(r[ye] || r[vl] || l === "link" && r.getAttribute("rel") === "stylesheet") && r.namespaceURI !== "http://www.w3.org/2000/svg") {
        var y = r.getAttribute(n) || "";
        y = l + y;
        var p = c.get(y);
        p ? p.push(r) : c.set(y, [r]);
      }
    }
    return c;
  }
  function Dm(l, n, u) {
    l = l.ownerDocument || l, l.head.insertBefore(
      u,
      n === "title" ? l.querySelector("head > title") : null
    );
  }
  function zv(l, n, u) {
    if (u === 1 || n.itemProp != null) return !1;
    switch (l) {
      case "meta":
      case "title":
        return !0;
      case "style":
        if (typeof n.precedence != "string" || typeof n.href != "string" || n.href === "")
          break;
        return !0;
      case "link":
        if (typeof n.rel != "string" || typeof n.href != "string" || n.href === "" || n.onLoad || n.onError)
          break;
        switch (n.rel) {
          case "stylesheet":
            return l = n.disabled, typeof n.precedence == "string" && l == null;
          default:
            return !0;
        }
      case "script":
        if (n.async && typeof n.async != "function" && typeof n.async != "symbol" && !n.onLoad && !n.onError && n.src && typeof n.src == "string")
          return !0;
    }
    return !1;
  }
  function zm(l) {
    return !(l.type === "stylesheet" && (l.state.loading & 3) === 0);
  }
  var ef = null;
  function Mv() {
  }
  function _v(l, n, u) {
    if (ef === null) throw Error(_(475));
    var c = ef;
    if (n.type === "stylesheet" && (typeof u.media != "string" || matchMedia(u.media).matches !== !1) && (n.state.loading & 4) === 0) {
      if (n.instance === null) {
        var s = Fo(u.href), r = l.querySelector(
          Io(s)
        );
        if (r) {
          l = r._p, l !== null && typeof l == "object" && typeof l.then == "function" && (c.count++, c = Hs.bind(c), l.then(c, c)), n.state.loading |= 4, n.instance = r, ol(r);
          return;
        }
        r = l.ownerDocument || l, u = Po(u), (s = Ft.get(s)) && kd(u, s), r = r.createElement("link"), ol(r);
        var y = r;
        y._p = new Promise(function(p, S) {
          y.onload = p, y.onerror = S;
        }), Ce(r, "link", u), n.instance = r;
      }
      c.stylesheets === null && (c.stylesheets = /* @__PURE__ */ new Map()), c.stylesheets.set(n, l), (l = n.state.preload) && (n.state.loading & 3) === 0 && (c.count++, n = Hs.bind(c), l.addEventListener("load", n), l.addEventListener("error", n));
    }
  }
  function Mm() {
    if (ef === null) throw Error(_(475));
    var l = ef;
    return l.stylesheets && l.count === 0 && Ns(l, l.stylesheets), 0 < l.count ? function(n) {
      var u = setTimeout(function() {
        if (l.stylesheets && Ns(l, l.stylesheets), l.unsuspend) {
          var c = l.unsuspend;
          l.unsuspend = null, c();
        }
      }, 6e4);
      return l.unsuspend = n, function() {
        l.unsuspend = null, clearTimeout(u);
      };
    } : null;
  }
  function Hs() {
    if (this.count--, this.count === 0) {
      if (this.stylesheets) Ns(this, this.stylesheets);
      else if (this.unsuspend) {
        var l = this.unsuspend;
        this.unsuspend = null, l();
      }
    }
  }
  var tf = null;
  function Ns(l, n) {
    l.stylesheets = null, l.unsuspend !== null && (l.count++, tf = /* @__PURE__ */ new Map(), n.forEach(xa, l), tf = null, Hs.call(l));
  }
  function xa(l, n) {
    if (!(n.state.loading & 4)) {
      var u = tf.get(l);
      if (u) var c = u.get(null);
      else {
        u = /* @__PURE__ */ new Map(), tf.set(l, u);
        for (var s = l.querySelectorAll(
          "link[data-precedence],style[data-precedence]"
        ), r = 0; r < s.length; r++) {
          var y = s[r];
          (y.nodeName === "LINK" || y.getAttribute("media") !== "not all") && (u.set(y.dataset.precedence, y), c = y);
        }
        c && u.set(null, c);
      }
      s = n.instance, y = s.getAttribute("data-precedence"), r = u.get(y) || c, r === c && u.set(null, s), u.set(y, s), this.count++, c = Hs.bind(this), s.addEventListener("load", c), s.addEventListener("error", c), r ? r.parentNode.insertBefore(s, r.nextSibling) : (l = l.nodeType === 9 ? l.head : l, l.insertBefore(s, l.firstChild)), n.state.loading |= 4;
    }
  }
  var ba = {
    $$typeof: it,
    Provider: null,
    Consumer: null,
    _currentValue: P,
    _currentValue2: P,
    _threadCount: 0
  };
  function Gg(l, n, u, c, s, r, y, p) {
    this.tag = 1, this.containerInfo = l, this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.next = this.pendingContext = this.context = this.cancelPendingCommit = null, this.callbackPriority = 0, this.expirationTimes = ve(-1), this.entangledLanes = this.shellSuspendCounter = this.errorRecoveryDisabledLanes = this.expiredLanes = this.warmLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = ve(0), this.hiddenUpdates = ve(null), this.identifierPrefix = c, this.onUncaughtError = s, this.onCaughtError = r, this.onRecoverableError = y, this.pooledCache = null, this.pooledCacheLanes = 0, this.formState = p, this.incompleteTransitions = /* @__PURE__ */ new Map();
  }
  function _m(l, n, u, c, s, r, y, p, S, H, K, $) {
    return l = new Gg(
      l,
      n,
      u,
      y,
      p,
      S,
      H,
      $
    ), n = 1, r === !0 && (n |= 24), r = oa(3, null, null, n), l.current = r, r.stateNode = l, n = Ao(), n.refCount++, l.pooledCache = n, n.refCount++, r.memoizedState = {
      element: c,
      isDehydrated: u,
      cache: n
    }, Gr(r), l;
  }
  function Um(l) {
    return l ? (l = mo, l) : mo;
  }
  function Cm(l, n, u, c, s, r) {
    s = Um(s), c.context === null ? c.context = s : c.pendingContext = s, c = sa(n), c.payload = { element: u }, r = r === void 0 ? null : r, r !== null && (c.callback = r), u = Vn(l, c, n), u !== null && (Ua(u, l, n), yc(u, l, n));
  }
  function xm(l, n) {
    if (l = l.memoizedState, l !== null && l.dehydrated !== null) {
      var u = l.retryLane;
      l.retryLane = u !== 0 && u < n ? u : n;
    }
  }
  function Wd(l, n) {
    xm(l, n), (l = l.alternate) && xm(l, n);
  }
  function Hm(l) {
    if (l.tag === 13) {
      var n = qn(l, 67108864);
      n !== null && Ua(n, l, 67108864), Wd(l, 67108864);
    }
  }
  var ws = !0;
  function Uv(l, n, u, c) {
    var s = O.T;
    O.T = null;
    var r = F.p;
    try {
      F.p = 2, Nm(l, n, u, c);
    } finally {
      F.p = r, O.T = s;
    }
  }
  function Cv(l, n, u, c) {
    var s = O.T;
    O.T = null;
    var r = F.p;
    try {
      F.p = 8, Nm(l, n, u, c);
    } finally {
      F.p = r, O.T = s;
    }
  }
  function Nm(l, n, u, c) {
    if (ws) {
      var s = Fd(c);
      if (s === null)
        Ia(
          l,
          n,
          c,
          Id,
          u
        ), qc(l, c);
      else if (Hv(
        s,
        l,
        n,
        u,
        c
      ))
        c.stopPropagation();
      else if (qc(l, c), n & 4 && -1 < xv.indexOf(l)) {
        for (; s !== null; ) {
          var r = Ki(s);
          if (r !== null)
            switch (r.tag) {
              case 3:
                if (r = r.stateNode, r.current.memoizedState.isDehydrated) {
                  var y = zl(r.pendingLanes);
                  if (y !== 0) {
                    var p = r;
                    for (p.pendingLanes |= 2, p.entangledLanes |= 2; y; ) {
                      var S = 1 << 31 - Dl(y);
                      p.entanglements[1] |= S, y &= ~S;
                    }
                    pa(r), (gt & 6) === 0 && (Md = pl() + 500, Es(0));
                  }
                }
                break;
              case 13:
                p = qn(r, 2), p !== null && Ua(p, r, 2), Uc(), Wd(r, 2);
            }
          if (r = Fd(c), r === null && Ia(
            l,
            n,
            c,
            Id,
            u
          ), r === s) break;
          s = r;
        }
        s !== null && c.stopPropagation();
      } else
        Ia(
          l,
          n,
          c,
          null,
          u
        );
    }
  }
  function Fd(l) {
    return l = pr(l), wm(l);
  }
  var Id = null;
  function wm(l) {
    if (Id = null, l = Ml(l), l !== null) {
      var n = Ee(l);
      if (n === null) l = null;
      else {
        var u = n.tag;
        if (u === 13) {
          if (l = Qe(n), l !== null) return l;
          l = null;
        } else if (u === 3) {
          if (n.stateNode.current.memoizedState.isDehydrated)
            return n.tag === 3 ? n.stateNode.containerInfo : null;
          l = null;
        } else n !== l && (l = null);
      }
    }
    return Id = l, null;
  }
  function qm(l) {
    switch (l) {
      case "beforetoggle":
      case "cancel":
      case "click":
      case "close":
      case "contextmenu":
      case "copy":
      case "cut":
      case "auxclick":
      case "dblclick":
      case "dragend":
      case "dragstart":
      case "drop":
      case "focusin":
      case "focusout":
      case "input":
      case "invalid":
      case "keydown":
      case "keypress":
      case "keyup":
      case "mousedown":
      case "mouseup":
      case "paste":
      case "pause":
      case "play":
      case "pointercancel":
      case "pointerdown":
      case "pointerup":
      case "ratechange":
      case "reset":
      case "resize":
      case "seeked":
      case "submit":
      case "toggle":
      case "touchcancel":
      case "touchend":
      case "touchstart":
      case "volumechange":
      case "change":
      case "selectionchange":
      case "textInput":
      case "compositionstart":
      case "compositionend":
      case "compositionupdate":
      case "beforeblur":
      case "afterblur":
      case "beforeinput":
      case "blur":
      case "fullscreenchange":
      case "focus":
      case "hashchange":
      case "popstate":
      case "select":
      case "selectstart":
        return 2;
      case "drag":
      case "dragenter":
      case "dragexit":
      case "dragleave":
      case "dragover":
      case "mousemove":
      case "mouseout":
      case "mouseover":
      case "pointermove":
      case "pointerout":
      case "pointerover":
      case "scroll":
      case "touchmove":
      case "wheel":
      case "mouseenter":
      case "mouseleave":
      case "pointerenter":
      case "pointerleave":
        return 8;
      case "message":
        switch (Iu()) {
          case ir:
            return 2;
          case Ke:
            return 8;
          case Mn:
          case eo:
            return 32;
          case gu:
            return 268435456;
          default:
            return 32;
        }
      default:
        return 32;
    }
  }
  var lf = !1, uu = null, Qu = null, Zu = null, qs = /* @__PURE__ */ new Map(), js = /* @__PURE__ */ new Map(), ji = [], xv = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(
    " "
  );
  function qc(l, n) {
    switch (l) {
      case "focusin":
      case "focusout":
        uu = null;
        break;
      case "dragenter":
      case "dragleave":
        Qu = null;
        break;
      case "mouseover":
      case "mouseout":
        Zu = null;
        break;
      case "pointerover":
      case "pointerout":
        qs.delete(n.pointerId);
        break;
      case "gotpointercapture":
      case "lostpointercapture":
        js.delete(n.pointerId);
    }
  }
  function jc(l, n, u, c, s, r) {
    return l === null || l.nativeEvent !== r ? (l = {
      blockedOn: n,
      domEventName: u,
      eventSystemFlags: c,
      nativeEvent: r,
      targetContainers: [s]
    }, n !== null && (n = Ki(n), n !== null && Hm(n)), l) : (l.eventSystemFlags |= c, n = l.targetContainers, s !== null && n.indexOf(s) === -1 && n.push(s), l);
  }
  function Hv(l, n, u, c, s) {
    switch (n) {
      case "focusin":
        return uu = jc(
          uu,
          l,
          n,
          u,
          c,
          s
        ), !0;
      case "dragenter":
        return Qu = jc(
          Qu,
          l,
          n,
          u,
          c,
          s
        ), !0;
      case "mouseover":
        return Zu = jc(
          Zu,
          l,
          n,
          u,
          c,
          s
        ), !0;
      case "pointerover":
        var r = s.pointerId;
        return qs.set(
          r,
          jc(
            qs.get(r) || null,
            l,
            n,
            u,
            c,
            s
          )
        ), !0;
      case "gotpointercapture":
        return r = s.pointerId, js.set(
          r,
          jc(
            js.get(r) || null,
            l,
            n,
            u,
            c,
            s
          )
        ), !0;
    }
    return !1;
  }
  function jm(l) {
    var n = Ml(l.target);
    if (n !== null) {
      var u = Ee(n);
      if (u !== null) {
        if (n = u.tag, n === 13) {
          if (n = Qe(u), n !== null) {
            l.blockedOn = n, xh(l.priority, function() {
              if (u.tag === 13) {
                var c = _a();
                c = ll(c);
                var s = qn(u, c);
                s !== null && Ua(s, u, c), Wd(u, c);
              }
            });
            return;
          }
        } else if (n === 3 && u.stateNode.current.memoizedState.isDehydrated) {
          l.blockedOn = u.tag === 3 ? u.stateNode.containerInfo : null;
          return;
        }
      }
    }
    l.blockedOn = null;
  }
  function Bs(l) {
    if (l.blockedOn !== null) return !1;
    for (var n = l.targetContainers; 0 < n.length; ) {
      var u = Fd(l.nativeEvent);
      if (u === null) {
        u = l.nativeEvent;
        var c = new u.constructor(
          u.type,
          u
        );
        Ii = c, u.target.dispatchEvent(c), Ii = null;
      } else
        return n = Ki(u), n !== null && Hm(n), l.blockedOn = u, !1;
      n.shift();
    }
    return !0;
  }
  function Ys(l, n, u) {
    Bs(l) && u.delete(n);
  }
  function af() {
    lf = !1, uu !== null && Bs(uu) && (uu = null), Qu !== null && Bs(Qu) && (Qu = null), Zu !== null && Bs(Zu) && (Zu = null), qs.forEach(Ys), js.forEach(Ys);
  }
  function Pd(l, n) {
    l.blockedOn === n && (l.blockedOn = null, lf || (lf = !0, U.unstable_scheduleCallback(
      U.unstable_NormalPriority,
      af
    )));
  }
  var Bc = null;
  function Bm(l) {
    Bc !== l && (Bc = l, U.unstable_scheduleCallback(
      U.unstable_NormalPriority,
      function() {
        Bc === l && (Bc = null);
        for (var n = 0; n < l.length; n += 3) {
          var u = l[n], c = l[n + 1], s = l[n + 2];
          if (typeof c != "function") {
            if (wm(c || u) === null)
              continue;
            break;
          }
          var r = Ki(u);
          r !== null && (l.splice(n, 3), n -= 3, id(
            r,
            {
              pending: !0,
              data: s,
              method: u.method,
              action: c
            },
            c,
            s
          ));
        }
      }
    ));
  }
  function iu(l) {
    function n(S) {
      return Pd(S, l);
    }
    uu !== null && Pd(uu, l), Qu !== null && Pd(Qu, l), Zu !== null && Pd(Zu, l), qs.forEach(n), js.forEach(n);
    for (var u = 0; u < ji.length; u++) {
      var c = ji[u];
      c.blockedOn === l && (c.blockedOn = null);
    }
    for (; 0 < ji.length && (u = ji[0], u.blockedOn === null); )
      jm(u), u.blockedOn === null && ji.shift();
    if (u = (l.ownerDocument || l).$$reactFormReplay, u != null)
      for (c = 0; c < u.length; c += 3) {
        var s = u[c], r = u[c + 1], y = s[kl] || null;
        if (typeof r == "function")
          y || Bm(u);
        else if (y) {
          var p = null;
          if (r && r.hasAttribute("formAction")) {
            if (s = r, y = r[kl] || null)
              p = y.formAction;
            else if (wm(s) !== null) continue;
          } else p = y.action;
          typeof p == "function" ? u[c + 1] = p : (u.splice(c, 3), c -= 3), Bm(u);
        }
      }
  }
  function Ym(l) {
    this._internalRoot = l;
  }
  eh.prototype.render = Ym.prototype.render = function(l) {
    var n = this._internalRoot;
    if (n === null) throw Error(_(409));
    var u = n.current, c = _a();
    Cm(u, c, l, n, null, null);
  }, eh.prototype.unmount = Ym.prototype.unmount = function() {
    var l = this._internalRoot;
    if (l !== null) {
      this._internalRoot = null;
      var n = l.containerInfo;
      Cm(l.current, 2, null, l, null, null), Uc(), n[ao] = null;
    }
  };
  function eh(l) {
    this._internalRoot = l;
  }
  eh.prototype.unstable_scheduleHydration = function(l) {
    if (l) {
      var n = or();
      l = { blockedOn: null, target: l, priority: n };
      for (var u = 0; u < ji.length && n !== 0 && n < ji[u].priority; u++) ;
      ji.splice(u, 0, l), u === 0 && jm(l);
    }
  };
  var Gm = W.version;
  if (Gm !== "19.1.1")
    throw Error(
      _(
        527,
        Gm,
        "19.1.1"
      )
    );
  F.findDOMNode = function(l) {
    var n = l._reactInternals;
    if (n === void 0)
      throw typeof l.render == "function" ? Error(_(188)) : (l = Object.keys(l).join(","), Error(_(268, l)));
    return l = x(n), l = l !== null ? w(l) : null, l = l === null ? null : l.stateNode, l;
  };
  var ea = {
    bundleType: 0,
    version: "19.1.1",
    rendererPackageName: "react-dom",
    currentDispatcherRef: O,
    reconcilerVersion: "19.1.1"
  };
  if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u") {
    var Gs = __REACT_DEVTOOLS_GLOBAL_HOOK__;
    if (!Gs.isDisabled && Gs.supportsFiber)
      try {
        Pu = Gs.inject(
          ea
        ), Ol = Gs;
      } catch {
      }
  }
  return Ap.createRoot = function(l, n) {
    if (!he(l)) throw Error(_(299));
    var u = !1, c = "", s = Co, r = jy, y = rs, p = null;
    return n != null && (n.unstable_strictMode === !0 && (u = !0), n.identifierPrefix !== void 0 && (c = n.identifierPrefix), n.onUncaughtError !== void 0 && (s = n.onUncaughtError), n.onCaughtError !== void 0 && (r = n.onCaughtError), n.onRecoverableError !== void 0 && (y = n.onRecoverableError), n.unstable_transitionCallbacks !== void 0 && (p = n.unstable_transitionCallbacks)), n = _m(
      l,
      1,
      !1,
      null,
      null,
      u,
      c,
      s,
      r,
      y,
      p,
      null
    ), l[ao] = n.current, Tm(l), new Ym(n);
  }, Ap.hydrateRoot = function(l, n, u) {
    if (!he(l)) throw Error(_(299));
    var c = !1, s = "", r = Co, y = jy, p = rs, S = null, H = null;
    return u != null && (u.unstable_strictMode === !0 && (c = !0), u.identifierPrefix !== void 0 && (s = u.identifierPrefix), u.onUncaughtError !== void 0 && (r = u.onUncaughtError), u.onCaughtError !== void 0 && (y = u.onCaughtError), u.onRecoverableError !== void 0 && (p = u.onRecoverableError), u.unstable_transitionCallbacks !== void 0 && (S = u.unstable_transitionCallbacks), u.formState !== void 0 && (H = u.formState)), n = _m(
      l,
      1,
      !0,
      n,
      u ?? null,
      c,
      s,
      r,
      y,
      p,
      S,
      H
    ), n.context = Um(null), u = n.current, c = _a(), c = ll(c), s = sa(c), s.callback = null, Vn(u, s, c), u = c, n.current.lanes = u, Ne(n, u), pa(n), l[ao] = n.current, Tm(l), new eh(n);
  }, Ap.version = "19.1.1", Ap;
}
var Rp = {}, eS;
function zT() {
  return eS || (eS = 1, It.env.NODE_ENV !== "production" && function() {
    function U(e, t) {
      for (e = e.memoizedState; e !== null && 0 < t; )
        e = e.next, t--;
      return e;
    }
    function W(e, t, a, i) {
      if (a >= t.length) return i;
      var o = t[a], f = we(e) ? e.slice() : Je({}, e);
      return f[o] = W(e[o], t, a + 1, i), f;
    }
    function ge(e, t, a) {
      if (t.length !== a.length)
        console.warn("copyWithRename() expects paths of the same length");
      else {
        for (var i = 0; i < a.length - 1; i++)
          if (t[i] !== a[i]) {
            console.warn(
              "copyWithRename() expects paths to be the same except for the deepest key"
            );
            return;
          }
        return _(e, t, a, 0);
      }
    }
    function _(e, t, a, i) {
      var o = t[i], f = we(e) ? e.slice() : Je({}, e);
      return i + 1 === t.length ? (f[a[i]] = f[o], we(f) ? f.splice(o, 1) : delete f[o]) : f[o] = _(
        e[o],
        t,
        a,
        i + 1
      ), f;
    }
    function he(e, t, a) {
      var i = t[a], o = we(e) ? e.slice() : Je({}, e);
      return a + 1 === t.length ? (we(o) ? o.splice(i, 1) : delete o[i], o) : (o[i] = he(e[i], t, a + 1), o);
    }
    function Ee() {
      return !1;
    }
    function Qe() {
      return null;
    }
    function et() {
    }
    function x() {
      console.error(
        "Do not call Hooks inside useEffect(...), useMemo(...), or other built-in Hooks. You can only call Hooks at the top level of your React function. For more information, see https://react.dev/link/rules-of-hooks"
      );
    }
    function w() {
      console.error(
        "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
      );
    }
    function te() {
    }
    function G(e) {
      var t = [];
      return e.forEach(function(a) {
        t.push(a);
      }), t.sort().join(", ");
    }
    function z(e, t, a, i) {
      return new wf(e, t, a, i);
    }
    function ae(e, t) {
      e.context === nf && (Tt(e.current, 2, t, e, null, null), Rc());
    }
    function je(e, t) {
      if (ou !== null) {
        var a = t.staleFamilies;
        t = t.updatedFamilies, xo(), Nf(
          e.current,
          t,
          a
        ), Rc();
      }
    }
    function At(e) {
      ou = e;
    }
    function ke(e) {
      return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11);
    }
    function nt(e) {
      var t = e, a = e;
      if (e.alternate) for (; t.return; ) t = t.return;
      else {
        e = t;
        do
          t = e, (t.flags & 4098) !== 0 && (a = t.return), e = t.return;
        while (e);
      }
      return t.tag === 3 ? a : null;
    }
    function el(e) {
      if (e.tag === 13) {
        var t = e.memoizedState;
        if (t === null && (e = e.alternate, e !== null && (t = e.memoizedState)), t !== null) return t.dehydrated;
      }
      return null;
    }
    function it(e) {
      if (nt(e) !== e)
        throw Error("Unable to find node on an unmounted component.");
    }
    function Nt(e) {
      var t = e.alternate;
      if (!t) {
        if (t = nt(e), t === null)
          throw Error("Unable to find node on an unmounted component.");
        return t !== e ? null : e;
      }
      for (var a = e, i = t; ; ) {
        var o = a.return;
        if (o === null) break;
        var f = o.alternate;
        if (f === null) {
          if (i = o.return, i !== null) {
            a = i;
            continue;
          }
          break;
        }
        if (o.child === f.child) {
          for (f = o.child; f; ) {
            if (f === a) return it(o), e;
            if (f === i) return it(o), t;
            f = f.sibling;
          }
          throw Error("Unable to find node on an unmounted component.");
        }
        if (a.return !== i.return) a = o, i = f;
        else {
          for (var d = !1, h = o.child; h; ) {
            if (h === a) {
              d = !0, a = o, i = f;
              break;
            }
            if (h === i) {
              d = !0, i = o, a = f;
              break;
            }
            h = h.sibling;
          }
          if (!d) {
            for (h = f.child; h; ) {
              if (h === a) {
                d = !0, a = f, i = o;
                break;
              }
              if (h === i) {
                d = !0, i = f, a = o;
                break;
              }
              h = h.sibling;
            }
            if (!d)
              throw Error(
                "Child was not found in either parent set. This indicates a bug in React related to the return pointer. Please file an issue."
              );
          }
        }
        if (a.alternate !== i)
          throw Error(
            "Return fibers should always be each others' alternates. This error is likely caused by a bug in React. Please file an issue."
          );
      }
      if (a.tag !== 3)
        throw Error("Unable to find node on an unmounted component.");
      return a.stateNode.current === a ? e : t;
    }
    function se(e) {
      var t = e.tag;
      if (t === 5 || t === 26 || t === 27 || t === 6) return e;
      for (e = e.child; e !== null; ) {
        if (t = se(e), t !== null) return t;
        e = e.sibling;
      }
      return null;
    }
    function Ue(e) {
      return e === null || typeof e != "object" ? null : (e = Am && e[Am] || e["@@iterator"], typeof e == "function" ? e : null);
    }
    function De(e) {
      if (e == null) return null;
      if (typeof e == "function")
        return e.$$typeof === Gd ? null : e.displayName || e.name || null;
      if (typeof e == "string") return e;
      switch (e) {
        case Le:
          return "Fragment";
        case Ko:
          return "Profiler";
        case Zo:
          return "StrictMode";
        case Jo:
          return "Suspense";
        case Ci:
          return "SuspenseList";
        case Em:
          return "Activity";
      }
      if (typeof e == "object")
        switch (typeof e.tag == "number" && console.error(
          "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
        ), e.$$typeof) {
          case Nc:
            return "Portal";
          case Ia:
            return (e.displayName || "Context") + ".Provider";
          case Yd:
            return (e._context.displayName || "Context") + ".Consumer";
          case Yu:
            var t = e.render;
            return e = e.displayName, e || (e = t.displayName || t.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
          case Ms:
            return t = e.displayName || null, t !== null ? t : De(e.type) || "Memo";
          case Ca:
            t = e._payload, e = e._init;
            try {
              return De(e(t));
            } catch {
            }
        }
      return null;
    }
    function bt(e) {
      return typeof e.tag == "number" ? re(e) : typeof e.name == "string" ? e.name : null;
    }
    function re(e) {
      var t = e.type;
      switch (e.tag) {
        case 31:
          return "Activity";
        case 24:
          return "Cache";
        case 9:
          return (t._context.displayName || "Context") + ".Consumer";
        case 10:
          return (t.displayName || "Context") + ".Provider";
        case 18:
          return "DehydratedFragment";
        case 11:
          return e = t.render, e = e.displayName || e.name || "", t.displayName || (e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef");
        case 7:
          return "Fragment";
        case 26:
        case 27:
        case 5:
          return t;
        case 4:
          return "Portal";
        case 3:
          return "Root";
        case 6:
          return "Text";
        case 16:
          return De(t);
        case 8:
          return t === Zo ? "StrictMode" : "Mode";
        case 22:
          return "Offscreen";
        case 12:
          return "Profiler";
        case 21:
          return "Scope";
        case 13:
          return "Suspense";
        case 19:
          return "SuspenseList";
        case 25:
          return "TracingMarker";
        case 1:
        case 0:
        case 14:
        case 15:
          if (typeof t == "function")
            return t.displayName || t.name || null;
          if (typeof t == "string") return t;
          break;
        case 29:
          if (t = e._debugInfo, t != null) {
            for (var a = t.length - 1; 0 <= a; a--)
              if (typeof t[a].name == "string") return t[a].name;
          }
          if (e.return !== null)
            return re(e.return);
      }
      return null;
    }
    function Rt(e) {
      return { current: e };
    }
    function Se(e, t) {
      0 > Pa ? console.error("Unexpected pop.") : (t !== Us[Pa] && console.error("Unexpected Fiber popped."), e.current = _s[Pa], _s[Pa] = null, Us[Pa] = null, Pa--);
    }
    function ze(e, t, a) {
      Pa++, _s[Pa] = e.current, Us[Pa] = a, e.current = t;
    }
    function Ot(e) {
      return e === null && console.error(
        "Expected host context to exist. This error is likely caused by a bug in React. Please file an issue."
      ), e;
    }
    function Gt(e, t) {
      ze(au, t, e), ze(ko, e, e), ze(Gu, null, e);
      var a = t.nodeType;
      switch (a) {
        case 9:
        case 11:
          a = a === 9 ? "#document" : "#fragment", t = (t = t.documentElement) && (t = t.namespaceURI) ? St(t) : kc;
          break;
        default:
          if (a = t.tagName, t = t.namespaceURI)
            t = St(t), t = ya(
              t,
              a
            );
          else
            switch (a) {
              case "svg":
                t = Mh;
                break;
              case "math":
                t = fg;
                break;
              default:
                t = kc;
            }
      }
      a = a.toLowerCase(), a = jh(null, a), a = {
        context: t,
        ancestorInfo: a
      }, Se(Gu, e), ze(Gu, a, e);
    }
    function pt(e) {
      Se(Gu, e), Se(ko, e), Se(au, e);
    }
    function O() {
      return Ot(Gu.current);
    }
    function F(e) {
      e.memoizedState !== null && ze($o, e, e);
      var t = Ot(Gu.current), a = e.type, i = ya(t.context, a);
      a = jh(t.ancestorInfo, a), i = { context: i, ancestorInfo: a }, t !== i && (ze(ko, e, e), ze(Gu, i, e));
    }
    function P(e) {
      ko.current === e && (Se(Gu, e), Se(ko, e)), $o.current === e && (Se($o, e), gp._currentValue = nr);
    }
    function be(e) {
      return typeof Symbol == "function" && Symbol.toStringTag && e[Symbol.toStringTag] || e.constructor.name || "Object";
    }
    function g(e) {
      try {
        return j(e), !1;
      } catch {
        return !0;
      }
    }
    function j(e) {
      return "" + e;
    }
    function J(e, t) {
      if (g(e))
        return console.error(
          "The provided `%s` attribute is an unsupported type %s. This value must be coerced to a string before using it here.",
          t,
          be(e)
        ), j(e);
    }
    function I(e, t) {
      if (g(e))
        return console.error(
          "The provided `%s` CSS property is an unsupported type %s. This value must be coerced to a string before using it here.",
          t,
          be(e)
        ), j(e);
    }
    function ce(e) {
      if (g(e))
        return console.error(
          "Form field values (value, checked, defaultValue, or defaultChecked props) must be strings, not %s. This value must be coerced to a string before using it here.",
          be(e)
        ), j(e);
    }
    function Oe(e) {
      if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u") return !1;
      var t = __REACT_DEVTOOLS_GLOBAL_HOOK__;
      if (t.isDisabled) return !0;
      if (!t.supportsFiber)
        return console.error(
          "The installed version of React DevTools is too old and will not work with the current version of React. Please update React DevTools. https://react.dev/link/react-devtools"
        ), !0;
      try {
        Hi = t.inject(e), wl = t;
      } catch (a) {
        console.error("React instrumentation encountered an error: %s.", a);
      }
      return !!t.checkDCE;
    }
    function oe(e) {
      if (typeof Yg == "function" && Tn(e), wl && typeof wl.setStrictMode == "function")
        try {
          wl.setStrictMode(Hi, e);
        } catch (t) {
          va || (va = !0, console.error(
            "React instrumentation encountered an error: %s",
            t
          ));
        }
    }
    function il(e) {
      fe = e;
    }
    function He() {
      fe !== null && typeof fe.markCommitStopped == "function" && fe.markCommitStopped();
    }
    function wt(e) {
      fe !== null && typeof fe.markComponentRenderStarted == "function" && fe.markComponentRenderStarted(e);
    }
    function na() {
      fe !== null && typeof fe.markComponentRenderStopped == "function" && fe.markComponentRenderStopped();
    }
    function Dn(e) {
      fe !== null && typeof fe.markRenderStarted == "function" && fe.markRenderStarted(e);
    }
    function Zi() {
      fe !== null && typeof fe.markRenderStopped == "function" && fe.markRenderStopped();
    }
    function zn(e, t) {
      fe !== null && typeof fe.markStateUpdateScheduled == "function" && fe.markStateUpdateScheduled(e, t);
    }
    function Pc(e) {
      return e >>>= 0, e === 0 ? 32 : 31 - (Xd(e) / Vu | 0) | 0;
    }
    function bf(e) {
      if (e & 1) return "SyncHydrationLane";
      if (e & 2) return "Sync";
      if (e & 4) return "InputContinuousHydration";
      if (e & 8) return "InputContinuous";
      if (e & 16) return "DefaultHydration";
      if (e & 32) return "Default";
      if (e & 128) return "TransitionHydration";
      if (e & 4194048) return "Transition";
      if (e & 62914560) return "Retry";
      if (e & 67108864) return "SelectiveHydration";
      if (e & 134217728) return "IdleHydration";
      if (e & 268435456) return "Idle";
      if (e & 536870912) return "Offscreen";
      if (e & 1073741824) return "Deferred";
    }
    function tl(e) {
      var t = e & 42;
      if (t !== 0) return t;
      switch (e & -e) {
        case 1:
          return 1;
        case 2:
          return 2;
        case 4:
          return 4;
        case 8:
          return 8;
        case 16:
          return 16;
        case 32:
          return 32;
        case 64:
          return 64;
        case 128:
          return 128;
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
        case 1048576:
        case 2097152:
          return e & 4194048;
        case 4194304:
        case 8388608:
        case 16777216:
        case 33554432:
          return e & 62914560;
        case 67108864:
          return 67108864;
        case 134217728:
          return 134217728;
        case 268435456:
          return 268435456;
        case 536870912:
          return 536870912;
        case 1073741824:
          return 0;
        default:
          return console.error(
            "Should have found matching lanes. This is a bug in React."
          ), e;
      }
    }
    function pl(e, t, a) {
      var i = e.pendingLanes;
      if (i === 0) return 0;
      var o = 0, f = e.suspendedLanes, d = e.pingedLanes;
      e = e.warmLanes;
      var h = i & 134217727;
      return h !== 0 ? (i = h & ~f, i !== 0 ? o = tl(i) : (d &= h, d !== 0 ? o = tl(d) : a || (a = h & ~e, a !== 0 && (o = tl(a))))) : (h = i & ~f, h !== 0 ? o = tl(h) : d !== 0 ? o = tl(d) : a || (a = i & ~e, a !== 0 && (o = tl(a)))), o === 0 ? 0 : t !== 0 && t !== o && (t & f) === 0 && (f = o & -o, a = t & -t, f >= a || f === 32 && (a & 4194048) !== 0) ? t : o;
    }
    function Iu(e, t) {
      return (e.pendingLanes & ~(e.suspendedLanes & ~e.pingedLanes) & t) === 0;
    }
    function ir(e, t) {
      switch (e) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 64:
          return t + 250;
        case 16:
        case 32:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
        case 1048576:
        case 2097152:
          return t + 5e3;
        case 4194304:
        case 8388608:
        case 16777216:
        case 33554432:
          return -1;
        case 67108864:
        case 134217728:
        case 268435456:
        case 536870912:
        case 1073741824:
          return -1;
        default:
          return console.error(
            "Should have found matching lanes. This is a bug in React."
          ), -1;
      }
    }
    function Ke() {
      var e = Qd;
      return Qd <<= 1, (Qd & 4194048) === 0 && (Qd = 256), e;
    }
    function Mn() {
      var e = Zd;
      return Zd <<= 1, (Zd & 62914560) === 0 && (Zd = 4194304), e;
    }
    function eo(e) {
      for (var t = [], a = 0; 31 > a; a++) t.push(e);
      return t;
    }
    function gu(e, t) {
      e.pendingLanes |= t, t !== 268435456 && (e.suspendedLanes = 0, e.pingedLanes = 0, e.warmLanes = 0);
    }
    function cr(e, t, a, i, o, f) {
      var d = e.pendingLanes;
      e.pendingLanes = a, e.suspendedLanes = 0, e.pingedLanes = 0, e.warmLanes = 0, e.expiredLanes &= a, e.entangledLanes &= a, e.errorRecoveryDisabledLanes &= a, e.shellSuspendCounter = 0;
      var h = e.entanglements, v = e.expirationTimes, b = e.hiddenUpdates;
      for (a = d & ~a; 0 < a; ) {
        var B = 31 - Ql(a), X = 1 << B;
        h[B] = 0, v[B] = -1;
        var N = b[B];
        if (N !== null)
          for (b[B] = null, B = 0; B < N.length; B++) {
            var Q = N[B];
            Q !== null && (Q.lane &= -536870913);
          }
        a &= ~X;
      }
      i !== 0 && Sf(e, i, 0), f !== 0 && o === 0 && e.tag !== 0 && (e.suspendedLanes |= f & ~(d & ~t));
    }
    function Sf(e, t, a) {
      e.pendingLanes |= t, e.suspendedLanes &= ~t;
      var i = 31 - Ql(t);
      e.entangledLanes |= t, e.entanglements[i] = e.entanglements[i] | 1073741824 | a & 4194090;
    }
    function Pu(e, t) {
      var a = e.entangledLanes |= t;
      for (e = e.entanglements; a; ) {
        var i = 31 - Ql(a), o = 1 << i;
        o & t | e[i] & t && (e[i] |= t), a &= ~o;
      }
    }
    function Ol(e) {
      switch (e) {
        case 2:
          e = 1;
          break;
        case 8:
          e = 4;
          break;
        case 32:
          e = 16;
          break;
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
        case 1048576:
        case 2097152:
        case 4194304:
        case 8388608:
        case 16777216:
        case 33554432:
          e = 128;
          break;
        case 268435456:
          e = 134217728;
          break;
        default:
          e = 0;
      }
      return e;
    }
    function Ba(e, t, a) {
      if (Ft)
        for (e = e.pendingUpdatersLaneMap; 0 < a; ) {
          var i = 31 - Ql(a), o = 1 << i;
          e[i].add(t), a &= ~o;
        }
    }
    function Dl(e, t) {
      if (Ft)
        for (var a = e.pendingUpdatersLaneMap, i = e.memoizedUpdaters; 0 < t; ) {
          var o = 31 - Ql(t);
          e = 1 << o, o = a[o], 0 < o.size && (o.forEach(function(f) {
            var d = f.alternate;
            d !== null && i.has(d) || i.add(f);
          }), o.clear()), t &= ~e;
        }
    }
    function to(e) {
      return e &= -e, ql < e ? En < e ? (e & 134217727) !== 0 ? Xu : Kd : En : ql;
    }
    function Tf() {
      var e = Ce.p;
      return e !== 0 ? e : (e = window.event, e === void 0 ? Xu : jd(e.type));
    }
    function lo(e, t) {
      var a = Ce.p;
      try {
        return Ce.p = e, t();
      } finally {
        Ce.p = a;
      }
    }
    function nn(e) {
      delete e[Zl], delete e[ga], delete e[Rm], delete e[Rv], delete e[Fo];
    }
    function ua(e) {
      var t = e[Zl];
      if (t) return t;
      for (var a = e.parentNode; a; ) {
        if (t = a[wi] || a[Zl]) {
          if (a = t.alternate, t.child !== null || a !== null && a.child !== null)
            for (e = Lo(e); e !== null; ) {
              if (a = e[Zl])
                return a;
              e = Lo(e);
            }
          return t;
        }
        e = a, a = e.parentNode;
      }
      return null;
    }
    function zl(e) {
      if (e = e[Zl] || e[wi]) {
        var t = e.tag;
        if (t === 5 || t === 6 || t === 13 || t === 26 || t === 27 || t === 3)
          return e;
      }
      return null;
    }
    function un(e) {
      var t = e.tag;
      if (t === 5 || t === 26 || t === 27 || t === 6)
        return e.stateNode;
      throw Error("getNodeFromInstance: Invalid argument.");
    }
    function m(e) {
      var t = e[Io];
      return t || (t = e[Io] = { hoistableStyles: /* @__PURE__ */ new Map(), hoistableScripts: /* @__PURE__ */ new Map() }), t;
    }
    function D(e) {
      e[Po] = !0;
    }
    function le(e, t) {
      ue(e, t), ue(e + "Capture", t);
    }
    function ue(e, t) {
      en[e] && console.error(
        "EventRegistry: More than one plugin attempted to publish the same registration name, `%s`.",
        e
      ), en[e] = t;
      var a = e.toLowerCase();
      for (wc[a] = e, e === "onDoubleClick" && (wc.ondblclick = e), e = 0; e < t.length; e++)
        Ov.add(t[e]);
    }
    function ve(e, t) {
      Dv[t.type] || t.onChange || t.onInput || t.readOnly || t.disabled || t.value == null || console.error(
        e === "select" ? "You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set `onChange`." : "You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set either `onChange` or `readOnly`."
      ), t.onChange || t.readOnly || t.disabled || t.checked == null || console.error(
        "You provided a `checked` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultChecked`. Otherwise, set either `onChange` or `readOnly`."
      );
    }
    function Ne(e) {
      return Lu.call($d, e) ? !0 : Lu.call(kd, e) ? !1 : Jd.test(e) ? $d[e] = !0 : (kd[e] = !0, console.error("Invalid attribute name: `%s`", e), !1);
    }
    function Ye(e, t, a) {
      if (Ne(t)) {
        if (!e.hasAttribute(t)) {
          switch (typeof a) {
            case "symbol":
            case "object":
              return a;
            case "function":
              return a;
            case "boolean":
              if (a === !1) return a;
          }
          return a === void 0 ? void 0 : null;
        }
        return e = e.getAttribute(t), e === "" && a === !0 ? !0 : (J(a, t), e === "" + a ? a : e);
      }
    }
    function ct(e, t, a) {
      if (Ne(t))
        if (a === null) e.removeAttribute(t);
        else {
          switch (typeof a) {
            case "undefined":
            case "function":
            case "symbol":
              e.removeAttribute(t);
              return;
            case "boolean":
              var i = t.toLowerCase().slice(0, 5);
              if (i !== "data-" && i !== "aria-") {
                e.removeAttribute(t);
                return;
              }
          }
          J(a, t), e.setAttribute(t, "" + a);
        }
    }
    function Be(e, t, a) {
      if (a === null) e.removeAttribute(t);
      else {
        switch (typeof a) {
          case "undefined":
          case "function":
          case "symbol":
          case "boolean":
            e.removeAttribute(t);
            return;
        }
        J(a, t), e.setAttribute(t, "" + a);
      }
    }
    function ll(e, t, a, i) {
      if (i === null) e.removeAttribute(a);
      else {
        switch (typeof i) {
          case "undefined":
          case "function":
          case "symbol":
          case "boolean":
            e.removeAttribute(a);
            return;
        }
        J(i, a), e.setAttributeNS(t, a, "" + i);
      }
    }
    function cn() {
    }
    function or() {
      if (qi === 0) {
        Om = console.log, Dm = console.info, zv = console.warn, zm = console.error, ef = console.group, Mv = console.groupCollapsed, _v = console.groupEnd;
        var e = {
          configurable: !0,
          enumerable: !0,
          value: cn,
          writable: !0
        };
        Object.defineProperties(console, {
          info: e,
          log: e,
          warn: e,
          error: e,
          group: e,
          groupCollapsed: e,
          groupEnd: e
        });
      }
      qi++;
    }
    function xh() {
      if (qi--, qi === 0) {
        var e = { configurable: !0, enumerable: !0, writable: !0 };
        Object.defineProperties(console, {
          log: Je({}, e, { value: Om }),
          info: Je({}, e, { value: Dm }),
          warn: Je({}, e, { value: zv }),
          error: Je({}, e, { value: zm }),
          group: Je({}, e, { value: ef }),
          groupCollapsed: Je({}, e, { value: Mv }),
          groupEnd: Je({}, e, { value: _v })
        });
      }
      0 > qi && console.error(
        "disabledDepth fell below zero. This is a bug in React. Please file an issue."
      );
    }
    function cl(e) {
      if (Mm === void 0)
        try {
          throw Error();
        } catch (a) {
          var t = a.stack.trim().match(/\n( *(at )?)/);
          Mm = t && t[1] || "", Hs = -1 < a.stack.indexOf(`
    at`) ? " (<anonymous>)" : -1 < a.stack.indexOf("@") ? "@unknown:0:0" : "";
        }
      return `
` + Mm + e + Hs;
    }
    function vl(e, t) {
      if (!e || tf) return "";
      var a = Ns.get(e);
      if (a !== void 0) return a;
      tf = !0, a = Error.prepareStackTrace, Error.prepareStackTrace = void 0;
      var i = null;
      i = L.H, L.H = null, or();
      try {
        var o = {
          DetermineComponentFrameRoot: function() {
            try {
              if (t) {
                var N = function() {
                  throw Error();
                };
                if (Object.defineProperty(N.prototype, "props", {
                  set: function() {
                    throw Error();
                  }
                }), typeof Reflect == "object" && Reflect.construct) {
                  try {
                    Reflect.construct(N, []);
                  } catch (me) {
                    var Q = me;
                  }
                  Reflect.construct(e, [], N);
                } else {
                  try {
                    N.call();
                  } catch (me) {
                    Q = me;
                  }
                  e.call(N.prototype);
                }
              } else {
                try {
                  throw Error();
                } catch (me) {
                  Q = me;
                }
                (N = e()) && typeof N.catch == "function" && N.catch(function() {
                });
              }
            } catch (me) {
              if (me && Q && typeof me.stack == "string")
                return [me.stack, Q.stack];
            }
            return [null, null];
          }
        };
        o.DetermineComponentFrameRoot.displayName = "DetermineComponentFrameRoot";
        var f = Object.getOwnPropertyDescriptor(
          o.DetermineComponentFrameRoot,
          "name"
        );
        f && f.configurable && Object.defineProperty(
          o.DetermineComponentFrameRoot,
          "name",
          { value: "DetermineComponentFrameRoot" }
        );
        var d = o.DetermineComponentFrameRoot(), h = d[0], v = d[1];
        if (h && v) {
          var b = h.split(`
`), B = v.split(`
`);
          for (d = f = 0; f < b.length && !b[f].includes(
            "DetermineComponentFrameRoot"
          ); )
            f++;
          for (; d < B.length && !B[d].includes(
            "DetermineComponentFrameRoot"
          ); )
            d++;
          if (f === b.length || d === B.length)
            for (f = b.length - 1, d = B.length - 1; 1 <= f && 0 <= d && b[f] !== B[d]; )
              d--;
          for (; 1 <= f && 0 <= d; f--, d--)
            if (b[f] !== B[d]) {
              if (f !== 1 || d !== 1)
                do
                  if (f--, d--, 0 > d || b[f] !== B[d]) {
                    var X = `
` + b[f].replace(
                      " at new ",
                      " at "
                    );
                    return e.displayName && X.includes("<anonymous>") && (X = X.replace("<anonymous>", e.displayName)), typeof e == "function" && Ns.set(e, X), X;
                  }
                while (1 <= f && 0 <= d);
              break;
            }
        }
      } finally {
        tf = !1, L.H = i, xh(), Error.prepareStackTrace = a;
      }
      return b = (b = e ? e.displayName || e.name : "") ? cl(b) : "", typeof e == "function" && Ns.set(e, b), b;
    }
    function kl(e) {
      var t = Error.prepareStackTrace;
      if (Error.prepareStackTrace = void 0, e = e.stack, Error.prepareStackTrace = t, e.startsWith(`Error: react-stack-top-frame
`) && (e = e.slice(29)), t = e.indexOf(`
`), t !== -1 && (e = e.slice(t + 1)), t = e.indexOf("react_stack_bottom_frame"), t !== -1 && (t = e.lastIndexOf(
        `
`,
        t
      )), t !== -1)
        e = e.slice(0, t);
      else return "";
      return e;
    }
    function ao(e) {
      switch (e.tag) {
        case 26:
        case 27:
        case 5:
          return cl(e.type);
        case 16:
          return cl("Lazy");
        case 13:
          return cl("Suspense");
        case 19:
          return cl("SuspenseList");
        case 0:
        case 15:
          return vl(e.type, !1);
        case 11:
          return vl(e.type.render, !1);
        case 1:
          return vl(e.type, !0);
        case 31:
          return cl("Activity");
        default:
          return "";
      }
    }
    function fr(e) {
      try {
        var t = "";
        do {
          t += ao(e);
          var a = e._debugInfo;
          if (a)
            for (var i = a.length - 1; 0 <= i; i--) {
              var o = a[i];
              if (typeof o.name == "string") {
                var f = t, d = o.env, h = cl(
                  o.name + (d ? " [" + d + "]" : "")
                );
                t = f + h;
              }
            }
          e = e.return;
        } while (e);
        return t;
      } catch (v) {
        return `
Error generating stack: ` + v.message + `
` + v.stack;
      }
    }
    function Dp(e) {
      return (e = e ? e.displayName || e.name : "") ? cl(e) : "";
    }
    function sr() {
      if (xa === null) return null;
      var e = xa._debugOwner;
      return e != null ? bt(e) : null;
    }
    function zp() {
      if (xa === null) return "";
      var e = xa;
      try {
        var t = "";
        switch (e.tag === 6 && (e = e.return), e.tag) {
          case 26:
          case 27:
          case 5:
            t += cl(e.type);
            break;
          case 13:
            t += cl("Suspense");
            break;
          case 19:
            t += cl("SuspenseList");
            break;
          case 31:
            t += cl("Activity");
            break;
          case 30:
          case 0:
          case 15:
          case 1:
            e._debugOwner || t !== "" || (t += Dp(
              e.type
            ));
            break;
          case 11:
            e._debugOwner || t !== "" || (t += Dp(
              e.type.render
            ));
        }
        for (; e; )
          if (typeof e.tag == "number") {
            var a = e;
            e = a._debugOwner;
            var i = a._debugStack;
            e && i && (typeof i != "string" && (a._debugStack = i = kl(i)), i !== "" && (t += `
` + i));
          } else if (e.debugStack != null) {
            var o = e.debugStack;
            (e = e.owner) && o && (t += `
` + kl(o));
          } else break;
        var f = t;
      } catch (d) {
        f = `
Error generating stack: ` + d.message + `
` + d.stack;
      }
      return f;
    }
    function ye(e, t, a, i, o, f, d) {
      var h = xa;
      Ef(e);
      try {
        return e !== null && e._debugTask ? e._debugTask.run(
          t.bind(null, a, i, o, f, d)
        ) : t(a, i, o, f, d);
      } finally {
        Ef(h);
      }
      throw Error(
        "runWithFiberInDEV should never be called in production. This is a bug in React."
      );
    }
    function Ef(e) {
      L.getCurrentStack = e === null ? null : zp, ba = !1, xa = e;
    }
    function Ml(e) {
      switch (typeof e) {
        case "bigint":
        case "boolean":
        case "number":
        case "string":
        case "undefined":
          return e;
        case "object":
          return ce(e), e;
        default:
          return "";
      }
    }
    function Ki(e) {
      var t = e.type;
      return (e = e.nodeName) && e.toLowerCase() === "input" && (t === "checkbox" || t === "radio");
    }
    function Af(e) {
      var t = Ki(e) ? "checked" : "value", a = Object.getOwnPropertyDescriptor(
        e.constructor.prototype,
        t
      );
      ce(e[t]);
      var i = "" + e[t];
      if (!e.hasOwnProperty(t) && typeof a < "u" && typeof a.get == "function" && typeof a.set == "function") {
        var o = a.get, f = a.set;
        return Object.defineProperty(e, t, {
          configurable: !0,
          get: function() {
            return o.call(this);
          },
          set: function(d) {
            ce(d), i = "" + d, f.call(this, d);
          }
        }), Object.defineProperty(e, t, {
          enumerable: a.enumerable
        }), {
          getValue: function() {
            return i;
          },
          setValue: function(d) {
            ce(d), i = "" + d;
          },
          stopTracking: function() {
            e._valueTracker = null, delete e[t];
          }
        };
      }
    }
    function bu(e) {
      e._valueTracker || (e._valueTracker = Af(e));
    }
    function ol(e) {
      if (!e) return !1;
      var t = e._valueTracker;
      if (!t) return !0;
      var a = t.getValue(), i = "";
      return e && (i = Ki(e) ? e.checked ? "true" : "false" : e.value), e = i, e !== a ? (t.setValue(e), !0) : !1;
    }
    function Rf(e) {
      if (e = e || (typeof document < "u" ? document : void 0), typeof e > "u") return null;
      try {
        return e.activeElement || e.body;
      } catch {
        return e.body;
      }
    }
    function Aa(e) {
      return e.replace(
        Gg,
        function(t) {
          return "\\" + t.charCodeAt(0).toString(16) + " ";
        }
      );
    }
    function ei(e, t) {
      t.checked === void 0 || t.defaultChecked === void 0 || Um || (console.error(
        "%s contains an input of type %s with both checked and defaultChecked props. Input elements must be either controlled or uncontrolled (specify either the checked prop, or the defaultChecked prop, but not both). Decide between using a controlled or uncontrolled input element and remove one of these props. More info: https://react.dev/link/controlled-components",
        sr() || "A component",
        t.type
      ), Um = !0), t.value === void 0 || t.defaultValue === void 0 || _m || (console.error(
        "%s contains an input of type %s with both value and defaultValue props. Input elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled input element and remove one of these props. More info: https://react.dev/link/controlled-components",
        sr() || "A component",
        t.type
      ), _m = !0);
    }
    function ti(e, t, a, i, o, f, d, h) {
      e.name = "", d != null && typeof d != "function" && typeof d != "symbol" && typeof d != "boolean" ? (J(d, "type"), e.type = d) : e.removeAttribute("type"), t != null ? d === "number" ? (t === 0 && e.value === "" || e.value != t) && (e.value = "" + Ml(t)) : e.value !== "" + Ml(t) && (e.value = "" + Ml(t)) : d !== "submit" && d !== "reset" || e.removeAttribute("value"), t != null ? rr(e, d, Ml(t)) : a != null ? rr(e, d, Ml(a)) : i != null && e.removeAttribute("value"), o == null && f != null && (e.defaultChecked = !!f), o != null && (e.checked = o && typeof o != "function" && typeof o != "symbol"), h != null && typeof h != "function" && typeof h != "symbol" && typeof h != "boolean" ? (J(h, "name"), e.name = "" + Ml(h)) : e.removeAttribute("name");
    }
    function Mp(e, t, a, i, o, f, d, h) {
      if (f != null && typeof f != "function" && typeof f != "symbol" && typeof f != "boolean" && (J(f, "type"), e.type = f), t != null || a != null) {
        if (!(f !== "submit" && f !== "reset" || t != null))
          return;
        a = a != null ? "" + Ml(a) : "", t = t != null ? "" + Ml(t) : a, h || t === e.value || (e.value = t), e.defaultValue = t;
      }
      i = i ?? o, i = typeof i != "function" && typeof i != "symbol" && !!i, e.checked = h ? e.checked : !!i, e.defaultChecked = !!i, d != null && typeof d != "function" && typeof d != "symbol" && typeof d != "boolean" && (J(d, "name"), e.name = d);
    }
    function rr(e, t, a) {
      t === "number" && Rf(e.ownerDocument) === e || e.defaultValue === "" + a || (e.defaultValue = "" + a);
    }
    function Hh(e, t) {
      t.value == null && (typeof t.children == "object" && t.children !== null ? Ds.Children.forEach(t.children, function(a) {
        a == null || typeof a == "string" || typeof a == "number" || typeof a == "bigint" || xm || (xm = !0, console.error(
          "Cannot infer the option value of complex children. Pass a `value` prop or use a plain string as children to <option>."
        ));
      }) : t.dangerouslySetInnerHTML == null || Wd || (Wd = !0, console.error(
        "Pass a `value` prop if you set dangerouslyInnerHTML so React knows which value should be selected."
      ))), t.selected == null || Cm || (console.error(
        "Use the `defaultValue` or `value` props on <select> instead of setting `selected` on <option>."
      ), Cm = !0);
    }
    function _p() {
      var e = sr();
      return e ? `

Check the render method of \`` + e + "`." : "";
    }
    function Su(e, t, a, i) {
      if (e = e.options, t) {
        t = {};
        for (var o = 0; o < a.length; o++)
          t["$" + a[o]] = !0;
        for (a = 0; a < e.length; a++)
          o = t.hasOwnProperty("$" + e[a].value), e[a].selected !== o && (e[a].selected = o), o && i && (e[a].defaultSelected = !0);
      } else {
        for (a = "" + Ml(a), t = null, o = 0; o < e.length; o++) {
          if (e[o].value === a) {
            e[o].selected = !0, i && (e[o].defaultSelected = !0);
            return;
          }
          t !== null || e[o].disabled || (t = e[o]);
        }
        t !== null && (t.selected = !0);
      }
    }
    function Of(e, t) {
      for (e = 0; e < ws.length; e++) {
        var a = ws[e];
        if (t[a] != null) {
          var i = we(t[a]);
          t.multiple && !i ? console.error(
            "The `%s` prop supplied to <select> must be an array if `multiple` is true.%s",
            a,
            _p()
          ) : !t.multiple && i && console.error(
            "The `%s` prop supplied to <select> must be a scalar value if `multiple` is false.%s",
            a,
            _p()
          );
        }
      }
      t.value === void 0 || t.defaultValue === void 0 || Hm || (console.error(
        "Select elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled select element and remove one of these props. More info: https://react.dev/link/controlled-components"
      ), Hm = !0);
    }
    function _n(e, t) {
      t.value === void 0 || t.defaultValue === void 0 || Uv || (console.error(
        "%s contains a textarea with both value and defaultValue props. Textarea elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled textarea and remove one of these props. More info: https://react.dev/link/controlled-components",
        sr() || "A component"
      ), Uv = !0), t.children != null && t.value == null && console.error(
        "Use the `defaultValue` or `value` props instead of setting children on <textarea>."
      );
    }
    function dr(e, t, a) {
      if (t != null && (t = "" + Ml(t), t !== e.value && (e.value = t), a == null)) {
        e.defaultValue !== t && (e.defaultValue = t);
        return;
      }
      e.defaultValue = a != null ? "" + Ml(a) : "";
    }
    function Nh(e, t, a, i) {
      if (t == null) {
        if (i != null) {
          if (a != null)
            throw Error(
              "If you supply `defaultValue` on a <textarea>, do not pass children."
            );
          if (we(i)) {
            if (1 < i.length)
              throw Error("<textarea> can only have at most one child.");
            i = i[0];
          }
          a = i;
        }
        a == null && (a = ""), t = a;
      }
      a = Ml(t), e.defaultValue = a, i = e.textContent, i === a && i !== "" && i !== null && (e.value = i);
    }
    function Ji(e, t) {
      return e.serverProps === void 0 && e.serverTail.length === 0 && e.children.length === 1 && 3 < e.distanceFromLeaf && e.distanceFromLeaf > 15 - t ? Ji(e.children[0], t) : e;
    }
    function $l(e) {
      return "  " + "  ".repeat(e);
    }
    function li(e) {
      return "+ " + "  ".repeat(e);
    }
    function ki(e) {
      return "- " + "  ".repeat(e);
    }
    function wh(e) {
      switch (e.tag) {
        case 26:
        case 27:
        case 5:
          return e.type;
        case 16:
          return "Lazy";
        case 13:
          return "Suspense";
        case 19:
          return "SuspenseList";
        case 0:
        case 15:
          return e = e.type, e.displayName || e.name || null;
        case 11:
          return e = e.type.render, e.displayName || e.name || null;
        case 1:
          return e = e.type, e.displayName || e.name || null;
        default:
          return null;
      }
    }
    function Gl(e, t) {
      return Cv.test(e) ? (e = JSON.stringify(e), e.length > t - 2 ? 8 > t ? '{"..."}' : "{" + e.slice(0, t - 7) + '..."}' : "{" + e + "}") : e.length > t ? 5 > t ? '{"..."}' : e.slice(0, t - 3) + "..." : e;
    }
    function Df(e, t, a) {
      var i = 120 - 2 * a;
      if (t === null)
        return li(a) + Gl(e, i) + `
`;
      if (typeof t == "string") {
        for (var o = 0; o < t.length && o < e.length && t.charCodeAt(o) === e.charCodeAt(o); o++) ;
        return o > i - 8 && 10 < o && (e = "..." + e.slice(o - 8), t = "..." + t.slice(o - 8)), li(a) + Gl(e, i) + `
` + ki(a) + Gl(t, i) + `
`;
      }
      return $l(a) + Gl(e, i) + `
`;
    }
    function qh(e) {
      return Object.prototype.toString.call(e).replace(/^\[object (.*)\]$/, function(t, a) {
        return a;
      });
    }
    function ai(e, t) {
      switch (typeof e) {
        case "string":
          return e = JSON.stringify(e), e.length > t ? 5 > t ? '"..."' : e.slice(0, t - 4) + '..."' : e;
        case "object":
          if (e === null) return "null";
          if (we(e)) return "[...]";
          if (e.$$typeof === Ui)
            return (t = De(e.type)) ? "<" + t + ">" : "<...>";
          var a = qh(e);
          if (a === "Object") {
            a = "", t -= 2;
            for (var i in e)
              if (e.hasOwnProperty(i)) {
                var o = JSON.stringify(i);
                if (o !== '"' + i + '"' && (i = o), t -= i.length - 2, o = ai(
                  e[i],
                  15 > t ? t : 15
                ), t -= o.length, 0 > t) {
                  a += a === "" ? "..." : ", ...";
                  break;
                }
                a += (a === "" ? "" : ",") + i + ":" + o;
              }
            return "{" + a + "}";
          }
          return a;
        case "function":
          return (t = e.displayName || e.name) ? "function " + t : "function";
        default:
          return String(e);
      }
    }
    function $i(e, t) {
      return typeof e != "string" || Cv.test(e) ? "{" + ai(e, t - 2) + "}" : e.length > t - 2 ? 5 > t ? '"..."' : '"' + e.slice(0, t - 5) + '..."' : '"' + e + '"';
    }
    function no(e, t, a) {
      var i = 120 - a.length - e.length, o = [], f;
      for (f in t)
        if (t.hasOwnProperty(f) && f !== "children") {
          var d = $i(
            t[f],
            120 - a.length - f.length - 1
          );
          i -= f.length + d.length + 2, o.push(f + "=" + d);
        }
      return o.length === 0 ? a + "<" + e + `>
` : 0 < i ? a + "<" + e + " " + o.join(" ") + `>
` : a + "<" + e + `
` + a + "  " + o.join(`
` + a + "  ") + `
` + a + `>
`;
    }
    function Eg(e, t, a) {
      var i = "", o = Je({}, t), f;
      for (f in e)
        if (e.hasOwnProperty(f)) {
          delete o[f];
          var d = 120 - 2 * a - f.length - 2, h = ai(e[f], d);
          t.hasOwnProperty(f) ? (d = ai(t[f], d), i += li(a) + f + ": " + h + `
`, i += ki(a) + f + ": " + d + `
`) : i += li(a) + f + ": " + h + `
`;
        }
      for (var v in o)
        o.hasOwnProperty(v) && (e = ai(
          o[v],
          120 - 2 * a - v.length - 2
        ), i += ki(a) + v + ": " + e + `
`);
      return i;
    }
    function Ya(e, t, a, i) {
      var o = "", f = /* @__PURE__ */ new Map();
      for (b in a)
        a.hasOwnProperty(b) && f.set(
          b.toLowerCase(),
          b
        );
      if (f.size === 1 && f.has("children"))
        o += no(
          e,
          t,
          $l(i)
        );
      else {
        for (var d in t)
          if (t.hasOwnProperty(d) && d !== "children") {
            var h = 120 - 2 * (i + 1) - d.length - 1, v = f.get(d.toLowerCase());
            if (v !== void 0) {
              f.delete(d.toLowerCase());
              var b = t[d];
              v = a[v];
              var B = $i(
                b,
                h
              );
              h = $i(
                v,
                h
              ), typeof b == "object" && b !== null && typeof v == "object" && v !== null && qh(b) === "Object" && qh(v) === "Object" && (2 < Object.keys(b).length || 2 < Object.keys(v).length || -1 < B.indexOf("...") || -1 < h.indexOf("...")) ? o += $l(i + 1) + d + `={{
` + Eg(
                b,
                v,
                i + 2
              ) + $l(i + 1) + `}}
` : (o += li(i + 1) + d + "=" + B + `
`, o += ki(i + 1) + d + "=" + h + `
`);
            } else
              o += $l(i + 1) + d + "=" + $i(t[d], h) + `
`;
          }
        f.forEach(function(X) {
          if (X !== "children") {
            var N = 120 - 2 * (i + 1) - X.length - 1;
            o += ki(i + 1) + X + "=" + $i(a[X], N) + `
`;
          }
        }), o = o === "" ? $l(i) + "<" + e + `>
` : $l(i) + "<" + e + `
` + o + $l(i) + `>
`;
      }
      return e = a.children, t = t.children, typeof e == "string" || typeof e == "number" || typeof e == "bigint" ? (f = "", (typeof t == "string" || typeof t == "number" || typeof t == "bigint") && (f = "" + t), o += Df(f, "" + e, i + 1)) : (typeof t == "string" || typeof t == "number" || typeof t == "bigint") && (o = e == null ? o + Df("" + t, null, i + 1) : o + Df("" + t, void 0, i + 1)), o;
    }
    function hr(e, t) {
      var a = wh(e);
      if (a === null) {
        for (a = "", e = e.child; e; )
          a += hr(e, t), e = e.sibling;
        return a;
      }
      return $l(t) + "<" + a + `>
`;
    }
    function yr(e, t) {
      var a = Ji(e, t);
      if (a !== e && (e.children.length !== 1 || e.children[0] !== a))
        return $l(t) + `...
` + yr(a, t + 1);
      a = "";
      var i = e.fiber._debugInfo;
      if (i)
        for (var o = 0; o < i.length; o++) {
          var f = i[o].name;
          typeof f == "string" && (a += $l(t) + "<" + f + `>
`, t++);
        }
      if (i = "", o = e.fiber.pendingProps, e.fiber.tag === 6)
        i = Df(o, e.serverProps, t), t++;
      else if (f = wh(e.fiber), f !== null)
        if (e.serverProps === void 0) {
          i = t;
          var d = 120 - 2 * i - f.length - 2, h = "";
          for (b in o)
            if (o.hasOwnProperty(b) && b !== "children") {
              var v = $i(o[b], 15);
              if (d -= b.length + v.length + 2, 0 > d) {
                h += " ...";
                break;
              }
              h += " " + b + "=" + v;
            }
          i = $l(i) + "<" + f + h + `>
`, t++;
        } else
          e.serverProps === null ? (i = no(
            f,
            o,
            li(t)
          ), t++) : typeof e.serverProps == "string" ? console.error(
            "Should not have matched a non HostText fiber to a Text node. This is a bug in React."
          ) : (i = Ya(
            f,
            o,
            e.serverProps,
            t
          ), t++);
      var b = "";
      for (o = e.fiber.child, f = 0; o && f < e.children.length; )
        d = e.children[f], d.fiber === o ? (b += yr(d, t), f++) : b += hr(o, t), o = o.sibling;
      for (o && 0 < e.children.length && (b += $l(t) + `...
`), o = e.serverTail, e.serverProps === null && t--, e = 0; e < o.length; e++)
        f = o[e], b = typeof f == "string" ? b + (ki(t) + Gl(f, 120 - 2 * t) + `
`) : b + no(
          f.type,
          f.props,
          ki(t)
        );
      return a + i + b;
    }
    function zf(e) {
      try {
        return `

` + yr(e, 0);
      } catch {
        return "";
      }
    }
    function Wi(e, t, a) {
      for (var i = t, o = null, f = 0; i; )
        i === e && (f = 0), o = {
          fiber: i,
          children: o !== null ? [o] : [],
          serverProps: i === t ? a : i === e ? null : void 0,
          serverTail: [],
          distanceFromLeaf: f
        }, f++, i = i.return;
      return o !== null ? zf(o).replaceAll(/^[+-]/gm, ">") : "";
    }
    function jh(e, t) {
      var a = Je({}, e || qm), i = { tag: t };
      return Fd.indexOf(t) !== -1 && (a.aTagInScope = null, a.buttonTagInScope = null, a.nobrTagInScope = null), Id.indexOf(t) !== -1 && (a.pTagInButtonScope = null), Nm.indexOf(t) !== -1 && t !== "address" && t !== "div" && t !== "p" && (a.listItemTagAutoclosing = null, a.dlItemTagAutoclosing = null), a.current = i, t === "form" && (a.formTag = i), t === "a" && (a.aTagInScope = i), t === "button" && (a.buttonTagInScope = i), t === "nobr" && (a.nobrTagInScope = i), t === "p" && (a.pTagInButtonScope = i), t === "li" && (a.listItemTagAutoclosing = i), (t === "dd" || t === "dt") && (a.dlItemTagAutoclosing = i), t === "#document" || t === "html" ? a.containerTagInScope = null : a.containerTagInScope || (a.containerTagInScope = i), e !== null || t !== "#document" && t !== "html" && t !== "body" ? a.implicitRootScope === !0 && (a.implicitRootScope = !1) : a.implicitRootScope = !0, a;
    }
    function Bh(e, t, a) {
      switch (t) {
        case "select":
          return e === "hr" || e === "option" || e === "optgroup" || e === "script" || e === "template" || e === "#text";
        case "optgroup":
          return e === "option" || e === "#text";
        case "option":
          return e === "#text";
        case "tr":
          return e === "th" || e === "td" || e === "style" || e === "script" || e === "template";
        case "tbody":
        case "thead":
        case "tfoot":
          return e === "tr" || e === "style" || e === "script" || e === "template";
        case "colgroup":
          return e === "col" || e === "template";
        case "table":
          return e === "caption" || e === "colgroup" || e === "tbody" || e === "tfoot" || e === "thead" || e === "style" || e === "script" || e === "template";
        case "head":
          return e === "base" || e === "basefont" || e === "bgsound" || e === "link" || e === "meta" || e === "title" || e === "noscript" || e === "noframes" || e === "style" || e === "script" || e === "template";
        case "html":
          if (a) break;
          return e === "head" || e === "body" || e === "frameset";
        case "frameset":
          return e === "frame";
        case "#document":
          if (!a) return e === "html";
      }
      switch (e) {
        case "h1":
        case "h2":
        case "h3":
        case "h4":
        case "h5":
        case "h6":
          return t !== "h1" && t !== "h2" && t !== "h3" && t !== "h4" && t !== "h5" && t !== "h6";
        case "rp":
        case "rt":
          return wm.indexOf(t) === -1;
        case "caption":
        case "col":
        case "colgroup":
        case "frameset":
        case "frame":
        case "tbody":
        case "td":
        case "tfoot":
        case "th":
        case "thead":
        case "tr":
          return t == null;
        case "head":
          return a || t === null;
        case "html":
          return a && t === "#document" || t === null;
        case "body":
          return a && (t === "#document" || t === "html") || t === null;
      }
      return !0;
    }
    function uo(e, t) {
      switch (e) {
        case "address":
        case "article":
        case "aside":
        case "blockquote":
        case "center":
        case "details":
        case "dialog":
        case "dir":
        case "div":
        case "dl":
        case "fieldset":
        case "figcaption":
        case "figure":
        case "footer":
        case "header":
        case "hgroup":
        case "main":
        case "menu":
        case "nav":
        case "ol":
        case "p":
        case "section":
        case "summary":
        case "ul":
        case "pre":
        case "listing":
        case "table":
        case "hr":
        case "xmp":
        case "h1":
        case "h2":
        case "h3":
        case "h4":
        case "h5":
        case "h6":
          return t.pTagInButtonScope;
        case "form":
          return t.formTag || t.pTagInButtonScope;
        case "li":
          return t.listItemTagAutoclosing;
        case "dd":
        case "dt":
          return t.dlItemTagAutoclosing;
        case "button":
          return t.buttonTagInScope;
        case "a":
          return t.aTagInScope;
        case "nobr":
          return t.nobrTagInScope;
      }
      return null;
    }
    function Up(e, t) {
      for (; e; ) {
        switch (e.tag) {
          case 5:
          case 26:
          case 27:
            if (e.type === t) return e;
        }
        e = e.return;
      }
      return null;
    }
    function mr(e, t) {
      t = t || qm;
      var a = t.current;
      if (t = (a = Bh(
        e,
        a && a.tag,
        t.implicitRootScope
      ) ? null : a) ? null : uo(e, t), t = a || t, !t) return !0;
      var i = t.tag;
      if (t = String(!!a) + "|" + e + "|" + i, lf[t]) return !1;
      lf[t] = !0;
      var o = (t = xa) ? Up(t.return, i) : null, f = t !== null && o !== null ? Wi(o, t, null) : "", d = "<" + e + ">";
      return a ? (a = "", i === "table" && e === "tr" && (a += " Add a <tbody>, <thead> or <tfoot> to your code to match the DOM tree generated by the browser."), console.error(
        `In HTML, %s cannot be a child of <%s>.%s
This will cause a hydration error.%s`,
        d,
        i,
        a,
        f
      )) : console.error(
        `In HTML, %s cannot be a descendant of <%s>.
This will cause a hydration error.%s`,
        d,
        i,
        f
      ), t && (e = t.return, o === null || e === null || o === e && e._debugOwner === t._debugOwner || ye(o, function() {
        console.error(
          `<%s> cannot contain a nested %s.
See this log for the ancestor stack trace.`,
          i,
          d
        );
      })), !1;
    }
    function Mf(e, t, a) {
      if (a || Bh("#text", t, !1))
        return !0;
      if (a = "#text|" + t, lf[a]) return !1;
      lf[a] = !0;
      var i = (a = xa) ? Up(a, t) : null;
      return a = a !== null && i !== null ? Wi(
        i,
        a,
        a.tag !== 6 ? { children: null } : null
      ) : "", /\S/.test(e) ? console.error(
        `In HTML, text nodes cannot be a child of <%s>.
This will cause a hydration error.%s`,
        t,
        a
      ) : console.error(
        `In HTML, whitespace text nodes cannot be a child of <%s>. Make sure you don't have any extra whitespace between tags on each line of your source code.
This will cause a hydration error.%s`,
        t,
        a
      ), !1;
    }
    function Fi(e, t) {
      if (t) {
        var a = e.firstChild;
        if (a && a === e.lastChild && a.nodeType === 3) {
          a.nodeValue = t;
          return;
        }
      }
      e.textContent = t;
    }
    function Ag(e) {
      return e.replace(ji, function(t, a) {
        return a.toUpperCase();
      });
    }
    function Cp(e, t, a) {
      var i = t.indexOf("--") === 0;
      i || (-1 < t.indexOf("-") ? qc.hasOwnProperty(t) && qc[t] || (qc[t] = !0, console.error(
        "Unsupported style property %s. Did you mean %s?",
        t,
        Ag(t.replace(js, "ms-"))
      )) : qs.test(t) ? qc.hasOwnProperty(t) && qc[t] || (qc[t] = !0, console.error(
        "Unsupported vendor-prefixed style property %s. Did you mean %s?",
        t,
        t.charAt(0).toUpperCase() + t.slice(1)
      )) : !xv.test(a) || jc.hasOwnProperty(a) && jc[a] || (jc[a] = !0, console.error(
        `Style property values shouldn't contain a semicolon. Try "%s: %s" instead.`,
        t,
        a.replace(xv, "")
      )), typeof a == "number" && (isNaN(a) ? Hv || (Hv = !0, console.error(
        "`NaN` is an invalid value for the `%s` css style property.",
        t
      )) : isFinite(a) || jm || (jm = !0, console.error(
        "`Infinity` is an invalid value for the `%s` css style property.",
        t
      )))), a == null || typeof a == "boolean" || a === "" ? i ? e.setProperty(t, "") : t === "float" ? e.cssFloat = "" : e[t] = "" : i ? e.setProperty(t, a) : typeof a != "number" || a === 0 || Bs.has(t) ? t === "float" ? e.cssFloat = a : (I(a, t), e[t] = ("" + a).trim()) : e[t] = a + "px";
    }
    function _f(e, t, a) {
      if (t != null && typeof t != "object")
        throw Error(
          "The `style` prop expects a mapping from style properties to values, not a string. For example, style={{marginRight: spacing + 'em'}} when using JSX."
        );
      if (t && Object.freeze(t), e = e.style, a != null) {
        if (t) {
          var i = {};
          if (a) {
            for (var o in a)
              if (a.hasOwnProperty(o) && !t.hasOwnProperty(o))
                for (var f = uu[o] || [o], d = 0; d < f.length; d++)
                  i[f[d]] = o;
          }
          for (var h in t)
            if (t.hasOwnProperty(h) && (!a || a[h] !== t[h]))
              for (o = uu[h] || [h], f = 0; f < o.length; f++)
                i[o[f]] = h;
          h = {};
          for (var v in t)
            for (o = uu[v] || [v], f = 0; f < o.length; f++)
              h[o[f]] = v;
          v = {};
          for (var b in i)
            if (o = i[b], (f = h[b]) && o !== f && (d = o + "," + f, !v[d])) {
              v[d] = !0, d = console;
              var B = t[o];
              d.error.call(
                d,
                "%s a style property during rerender (%s) when a conflicting property is set (%s) can lead to styling bugs. To avoid this, don't mix shorthand and non-shorthand properties for the same value; instead, replace the shorthand with separate values.",
                B == null || typeof B == "boolean" || B === "" ? "Removing" : "Updating",
                o,
                f
              );
            }
        }
        for (var X in a)
          !a.hasOwnProperty(X) || t != null && t.hasOwnProperty(X) || (X.indexOf("--") === 0 ? e.setProperty(X, "") : X === "float" ? e.cssFloat = "" : e[X] = "");
        for (var N in t)
          b = t[N], t.hasOwnProperty(N) && a[N] !== b && Cp(e, N, b);
      } else
        for (i in t)
          t.hasOwnProperty(i) && Cp(e, i, t[i]);
    }
    function Ii(e) {
      if (e.indexOf("-") === -1) return !1;
      switch (e) {
        case "annotation-xml":
        case "color-profile":
        case "font-face":
        case "font-face-src":
        case "font-face-uri":
        case "font-face-format":
        case "font-face-name":
        case "missing-glyph":
          return !1;
        default:
          return !0;
      }
    }
    function pr(e) {
      return Pd.get(e) || e;
    }
    function io(e, t) {
      if (Lu.call(iu, t) && iu[t])
        return !0;
      if (eh.test(t)) {
        if (e = "aria-" + t.slice(4).toLowerCase(), e = Bm.hasOwnProperty(e) ? e : null, e == null)
          return console.error(
            "Invalid ARIA attribute `%s`. ARIA attributes follow the pattern aria-* and must be lowercase.",
            t
          ), iu[t] = !0;
        if (t !== e)
          return console.error(
            "Invalid ARIA attribute `%s`. Did you mean `%s`?",
            t,
            e
          ), iu[t] = !0;
      }
      if (Ym.test(t)) {
        if (e = t.toLowerCase(), e = Bm.hasOwnProperty(e) ? e : null, e == null) return iu[t] = !0, !1;
        t !== e && (console.error(
          "Unknown ARIA attribute `%s`. Did you mean `%s`?",
          t,
          e
        ), iu[t] = !0);
      }
      return !0;
    }
    function co(e, t) {
      var a = [], i;
      for (i in t)
        io(e, i) || a.push(i);
      t = a.map(function(o) {
        return "`" + o + "`";
      }).join(", "), a.length === 1 ? console.error(
        "Invalid aria prop %s on <%s> tag. For details, see https://react.dev/link/invalid-aria-props",
        t,
        e
      ) : 1 < a.length && console.error(
        "Invalid aria props %s on <%s> tag. For details, see https://react.dev/link/invalid-aria-props",
        t,
        e
      );
    }
    function xp(e, t, a, i) {
      if (Lu.call(ea, t) && ea[t])
        return !0;
      var o = t.toLowerCase();
      if (o === "onfocusin" || o === "onfocusout")
        return console.error(
          "React uses onFocus and onBlur instead of onFocusIn and onFocusOut. All React events are normalized to bubble, so onFocusIn and onFocusOut are not needed/supported by React."
        ), ea[t] = !0;
      if (typeof a == "function" && (e === "form" && t === "action" || e === "input" && t === "formAction" || e === "button" && t === "formAction"))
        return !0;
      if (i != null) {
        if (e = i.possibleRegistrationNames, i.registrationNameDependencies.hasOwnProperty(t))
          return !0;
        if (i = e.hasOwnProperty(o) ? e[o] : null, i != null)
          return console.error(
            "Invalid event handler property `%s`. Did you mean `%s`?",
            t,
            i
          ), ea[t] = !0;
        if (Gs.test(t))
          return console.error(
            "Unknown event handler property `%s`. It will be ignored.",
            t
          ), ea[t] = !0;
      } else if (Gs.test(t))
        return l.test(t) && console.error(
          "Invalid event handler property `%s`. React events use the camelCase naming convention, for example `onClick`.",
          t
        ), ea[t] = !0;
      if (n.test(t) || u.test(t)) return !0;
      if (o === "innerhtml")
        return console.error(
          "Directly setting property `innerHTML` is not permitted. For more information, lookup documentation on `dangerouslySetInnerHTML`."
        ), ea[t] = !0;
      if (o === "aria")
        return console.error(
          "The `aria` attribute is reserved for future use in React. Pass individual `aria-` attributes instead."
        ), ea[t] = !0;
      if (o === "is" && a !== null && a !== void 0 && typeof a != "string")
        return console.error(
          "Received a `%s` for a string attribute `is`. If this is expected, cast the value to a string.",
          typeof a
        ), ea[t] = !0;
      if (typeof a == "number" && isNaN(a))
        return console.error(
          "Received NaN for the `%s` attribute. If this is expected, cast the value to a string.",
          t
        ), ea[t] = !0;
      if (Bc.hasOwnProperty(o)) {
        if (o = Bc[o], o !== t)
          return console.error(
            "Invalid DOM property `%s`. Did you mean `%s`?",
            t,
            o
          ), ea[t] = !0;
      } else if (t !== o)
        return console.error(
          "React does not recognize the `%s` prop on a DOM element. If you intentionally want it to appear in the DOM as a custom attribute, spell it as lowercase `%s` instead. If you accidentally passed it from a parent component, remove it from the DOM element.",
          t,
          o
        ), ea[t] = !0;
      switch (t) {
        case "dangerouslySetInnerHTML":
        case "children":
        case "style":
        case "suppressContentEditableWarning":
        case "suppressHydrationWarning":
        case "defaultValue":
        case "defaultChecked":
        case "innerHTML":
        case "ref":
          return !0;
        case "innerText":
        case "textContent":
          return !0;
      }
      switch (typeof a) {
        case "boolean":
          switch (t) {
            case "autoFocus":
            case "checked":
            case "multiple":
            case "muted":
            case "selected":
            case "contentEditable":
            case "spellCheck":
            case "draggable":
            case "value":
            case "autoReverse":
            case "externalResourcesRequired":
            case "focusable":
            case "preserveAlpha":
            case "allowFullScreen":
            case "async":
            case "autoPlay":
            case "controls":
            case "default":
            case "defer":
            case "disabled":
            case "disablePictureInPicture":
            case "disableRemotePlayback":
            case "formNoValidate":
            case "hidden":
            case "loop":
            case "noModule":
            case "noValidate":
            case "open":
            case "playsInline":
            case "readOnly":
            case "required":
            case "reversed":
            case "scoped":
            case "seamless":
            case "itemScope":
            case "capture":
            case "download":
            case "inert":
              return !0;
            default:
              return o = t.toLowerCase().slice(0, 5), o === "data-" || o === "aria-" ? !0 : (a ? console.error(
                'Received `%s` for a non-boolean attribute `%s`.\n\nIf you want to write it to the DOM, pass a string instead: %s="%s" or %s={value.toString()}.',
                a,
                t,
                t,
                a,
                t
              ) : console.error(
                'Received `%s` for a non-boolean attribute `%s`.\n\nIf you want to write it to the DOM, pass a string instead: %s="%s" or %s={value.toString()}.\n\nIf you used to conditionally omit it with %s={condition && value}, pass %s={condition ? value : undefined} instead.',
                a,
                t,
                t,
                a,
                t,
                t,
                t
              ), ea[t] = !0);
          }
        case "function":
        case "symbol":
          return ea[t] = !0, !1;
        case "string":
          if (a === "false" || a === "true") {
            switch (t) {
              case "checked":
              case "selected":
              case "multiple":
              case "muted":
              case "allowFullScreen":
              case "async":
              case "autoPlay":
              case "controls":
              case "default":
              case "defer":
              case "disabled":
              case "disablePictureInPicture":
              case "disableRemotePlayback":
              case "formNoValidate":
              case "hidden":
              case "loop":
              case "noModule":
              case "noValidate":
              case "open":
              case "playsInline":
              case "readOnly":
              case "required":
              case "reversed":
              case "scoped":
              case "seamless":
              case "itemScope":
              case "inert":
                break;
              default:
                return !0;
            }
            console.error(
              "Received the string `%s` for the boolean attribute `%s`. %s Did you mean %s={%s}?",
              a,
              t,
              a === "false" ? "The browser will interpret it as a truthy value." : 'Although this works, it will not work as expected if you pass the string "false".',
              t,
              a
            ), ea[t] = !0;
          }
      }
      return !0;
    }
    function Yh(e, t, a) {
      var i = [], o;
      for (o in t)
        xp(e, o, t[o], a) || i.push(o);
      t = i.map(function(f) {
        return "`" + f + "`";
      }).join(", "), i.length === 1 ? console.error(
        "Invalid value for prop %s on <%s> tag. Either remove it from the element, or pass a string or number value to keep it in the DOM. For details, see https://react.dev/link/attribute-behavior ",
        t,
        e
      ) : 1 < i.length && console.error(
        "Invalid values for props %s on <%s> tag. Either remove them from the element, or pass a string or number value to keep them in the DOM. For details, see https://react.dev/link/attribute-behavior ",
        t,
        e
      );
    }
    function oo(e) {
      return c.test("" + e) ? "javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')" : e;
    }
    function Pi(e) {
      return e = e.target || e.srcElement || window, e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e;
    }
    function Un(e) {
      var t = zl(e);
      if (t && (e = t.stateNode)) {
        var a = e[ga] || null;
        e: switch (e = t.stateNode, t.type) {
          case "input":
            if (ti(
              e,
              a.value,
              a.defaultValue,
              a.defaultValue,
              a.checked,
              a.defaultChecked,
              a.type,
              a.name
            ), t = a.name, a.type === "radio" && t != null) {
              for (a = e; a.parentNode; ) a = a.parentNode;
              for (J(t, "name"), a = a.querySelectorAll(
                'input[name="' + Aa(
                  "" + t
                ) + '"][type="radio"]'
              ), t = 0; t < a.length; t++) {
                var i = a[t];
                if (i !== e && i.form === e.form) {
                  var o = i[ga] || null;
                  if (!o)
                    throw Error(
                      "ReactDOMInput: Mixing React and non-React radio inputs with the same `name` is not supported."
                    );
                  ti(
                    i,
                    o.value,
                    o.defaultValue,
                    o.defaultValue,
                    o.checked,
                    o.defaultChecked,
                    o.type,
                    o.name
                  );
                }
              }
              for (t = 0; t < a.length; t++)
                i = a[t], i.form === e.form && ol(i);
            }
            break e;
          case "textarea":
            dr(e, a.value, a.defaultValue);
            break e;
          case "select":
            t = a.value, t != null && Su(e, !!a.multiple, t, !1);
        }
      }
    }
    function vr(e, t, a) {
      if (p) return e(t, a);
      p = !0;
      try {
        var i = e(t);
        return i;
      } finally {
        if (p = !1, (r !== null || y !== null) && (Rc(), r && (t = r, e = y, y = r = null, Un(t), e)))
          for (t = 0; t < e.length; t++) Un(e[t]);
      }
    }
    function Tu(e, t) {
      var a = e.stateNode;
      if (a === null) return null;
      var i = a[ga] || null;
      if (i === null) return null;
      a = i[t];
      e: switch (t) {
        case "onClick":
        case "onClickCapture":
        case "onDoubleClick":
        case "onDoubleClickCapture":
        case "onMouseDown":
        case "onMouseDownCapture":
        case "onMouseMove":
        case "onMouseMoveCapture":
        case "onMouseUp":
        case "onMouseUpCapture":
        case "onMouseEnter":
          (i = !i.disabled) || (e = e.type, i = !(e === "button" || e === "input" || e === "select" || e === "textarea")), e = !i;
          break e;
        default:
          e = !1;
      }
      if (e) return null;
      if (a && typeof a != "function")
        throw Error(
          "Expected `" + t + "` listener to be a function, instead got a value of `" + typeof a + "` type."
        );
      return a;
    }
    function Eu() {
      if (Y) return Y;
      var e, t = q, a = t.length, i, o = "value" in $ ? $.value : $.textContent, f = o.length;
      for (e = 0; e < a && t[e] === o[e]; e++) ;
      var d = a - e;
      for (i = 1; i <= d && t[a - i] === o[f - i]; i++) ;
      return Y = o.slice(e, 1 < i ? 1 - i : void 0);
    }
    function fo(e) {
      var t = e.keyCode;
      return "charCode" in e ? (e = e.charCode, e === 0 && t === 13 && (e = 13)) : e = t, e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0;
    }
    function ec() {
      return !0;
    }
    function Gh() {
      return !1;
    }
    function _l(e) {
      function t(a, i, o, f, d) {
        this._reactName = a, this._targetInst = o, this.type = i, this.nativeEvent = f, this.target = d, this.currentTarget = null;
        for (var h in e)
          e.hasOwnProperty(h) && (a = e[h], this[h] = a ? a(f) : f[h]);
        return this.isDefaultPrevented = (f.defaultPrevented != null ? f.defaultPrevented : f.returnValue === !1) ? ec : Gh, this.isPropagationStopped = Gh, this;
      }
      return Je(t.prototype, {
        preventDefault: function() {
          this.defaultPrevented = !0;
          var a = this.nativeEvent;
          a && (a.preventDefault ? a.preventDefault() : typeof a.returnValue != "unknown" && (a.returnValue = !1), this.isDefaultPrevented = ec);
        },
        stopPropagation: function() {
          var a = this.nativeEvent;
          a && (a.stopPropagation ? a.stopPropagation() : typeof a.cancelBubble != "unknown" && (a.cancelBubble = !0), this.isPropagationStopped = ec);
        },
        persist: function() {
        },
        isPersistent: ec
      }), t;
    }
    function gr(e) {
      var t = this.nativeEvent;
      return t.getModifierState ? t.getModifierState(e) : (e = yS[e]) ? !!t[e] : !1;
    }
    function br() {
      return gr;
    }
    function Wl(e, t) {
      switch (e) {
        case "keyup":
          return DS.indexOf(t.keyCode) !== -1;
        case "keydown":
          return t.keyCode !== Z0;
        case "keypress":
        case "mousedown":
        case "focusout":
          return !0;
        default:
          return !1;
      }
    }
    function ni(e) {
      return e = e.detail, typeof e == "object" && "data" in e ? e.data : null;
    }
    function Sr(e, t) {
      switch (e) {
        case "compositionend":
          return ni(t);
        case "keypress":
          return t.which !== J0 ? null : ($0 = !0, k0);
        case "textInput":
          return e = t.data, e === k0 && $0 ? null : e;
        default:
          return null;
      }
    }
    function Uf(e, t) {
      if (th)
        return e === "compositionend" || !Vg && Wl(e, t) ? (e = Eu(), Y = q = $ = null, th = !1, e) : null;
      switch (e) {
        case "paste":
          return null;
        case "keypress":
          if (!(t.ctrlKey || t.altKey || t.metaKey) || t.ctrlKey && t.altKey) {
            if (t.char && 1 < t.char.length)
              return t.char;
            if (t.which)
              return String.fromCharCode(t.which);
          }
          return null;
        case "compositionend":
          return K0 && t.locale !== "ko" ? null : t.data;
        default:
          return null;
      }
    }
    function Hp(e) {
      var t = e && e.nodeName && e.nodeName.toLowerCase();
      return t === "input" ? !!MS[e.type] : t === "textarea";
    }
    function Lh(e) {
      if (!S) return !1;
      e = "on" + e;
      var t = e in document;
      return t || (t = document.createElement("div"), t.setAttribute(e, "return;"), t = typeof t[e] == "function"), t;
    }
    function Tr(e, t, a, i) {
      r ? y ? y.push(i) : y = [i] : r = i, t = vs(t, "onChange"), 0 < t.length && (a = new Re(
        "onChange",
        "change",
        null,
        a,
        i
      ), e.push({ event: a, listeners: t }));
    }
    function Cf(e) {
      Fn(e, 0);
    }
    function tc(e) {
      var t = un(e);
      if (ol(t)) return e;
    }
    function Vh(e, t) {
      if (e === "change") return t;
    }
    function Np() {
      Vm && (Vm.detachEvent("onpropertychange", wp), Xm = Vm = null);
    }
    function wp(e) {
      if (e.propertyName === "value" && tc(Xm)) {
        var t = [];
        Tr(
          t,
          Xm,
          e,
          Pi(e)
        ), vr(Cf, t);
      }
    }
    function Rg(e, t, a) {
      e === "focusin" ? (Np(), Vm = t, Xm = a, Vm.attachEvent("onpropertychange", wp)) : e === "focusout" && Np();
    }
    function Xh(e) {
      if (e === "selectionchange" || e === "keyup" || e === "keydown")
        return tc(Xm);
    }
    function Og(e, t) {
      if (e === "click") return tc(t);
    }
    function Dg(e, t) {
      if (e === "input" || e === "change")
        return tc(t);
    }
    function zg(e, t) {
      return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t;
    }
    function xf(e, t) {
      if (Ha(e, t)) return !0;
      if (typeof e != "object" || e === null || typeof t != "object" || t === null)
        return !1;
      var a = Object.keys(e), i = Object.keys(t);
      if (a.length !== i.length) return !1;
      for (i = 0; i < a.length; i++) {
        var o = a[i];
        if (!Lu.call(t, o) || !Ha(e[o], t[o]))
          return !1;
      }
      return !0;
    }
    function qp(e) {
      for (; e && e.firstChild; ) e = e.firstChild;
      return e;
    }
    function Qh(e, t) {
      var a = qp(e);
      e = 0;
      for (var i; a; ) {
        if (a.nodeType === 3) {
          if (i = e + a.textContent.length, e <= t && i >= t)
            return { node: a, offset: t - e };
          e = i;
        }
        e: {
          for (; a; ) {
            if (a.nextSibling) {
              a = a.nextSibling;
              break e;
            }
            a = a.parentNode;
          }
          a = void 0;
        }
        a = qp(a);
      }
    }
    function jp(e, t) {
      return e && t ? e === t ? !0 : e && e.nodeType === 3 ? !1 : t && t.nodeType === 3 ? jp(e, t.parentNode) : "contains" in e ? e.contains(t) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(t) & 16) : !1 : !1;
    }
    function Bp(e) {
      e = e != null && e.ownerDocument != null && e.ownerDocument.defaultView != null ? e.ownerDocument.defaultView : window;
      for (var t = Rf(e.document); t instanceof e.HTMLIFrameElement; ) {
        try {
          var a = typeof t.contentWindow.location.href == "string";
        } catch {
          a = !1;
        }
        if (a) e = t.contentWindow;
        else break;
        t = Rf(e.document);
      }
      return t;
    }
    function Zh(e) {
      var t = e && e.nodeName && e.nodeName.toLowerCase();
      return t && (t === "input" && (e.type === "text" || e.type === "search" || e.type === "tel" || e.type === "url" || e.type === "password") || t === "textarea" || e.contentEditable === "true");
    }
    function Yp(e, t, a) {
      var i = a.window === a ? a.document : a.nodeType === 9 ? a : a.ownerDocument;
      Qg || lh == null || lh !== Rf(i) || (i = lh, "selectionStart" in i && Zh(i) ? i = { start: i.selectionStart, end: i.selectionEnd } : (i = (i.ownerDocument && i.ownerDocument.defaultView || window).getSelection(), i = {
        anchorNode: i.anchorNode,
        anchorOffset: i.anchorOffset,
        focusNode: i.focusNode,
        focusOffset: i.focusOffset
      }), Qm && xf(Qm, i) || (Qm = i, i = vs(Xg, "onSelect"), 0 < i.length && (t = new Re(
        "onSelect",
        "select",
        null,
        t,
        a
      ), e.push({ event: t, listeners: i }), t.target = lh)));
    }
    function Au(e, t) {
      var a = {};
      return a[e.toLowerCase()] = t.toLowerCase(), a["Webkit" + e] = "webkit" + t, a["Moz" + e] = "moz" + t, a;
    }
    function lc(e) {
      if (Zg[e]) return Zg[e];
      if (!ah[e]) return e;
      var t = ah[e], a;
      for (a in t)
        if (t.hasOwnProperty(a) && a in F0)
          return Zg[e] = t[a];
      return e;
    }
    function on(e, t) {
      l1.set(e, t), le(t, [e]);
    }
    function Ra(e, t) {
      if (typeof e == "object" && e !== null) {
        var a = Jg.get(e);
        return a !== void 0 ? a : (t = {
          value: e,
          source: t,
          stack: fr(t)
        }, Jg.set(e, t), t);
      }
      return {
        value: e,
        source: t,
        stack: fr(t)
      };
    }
    function Hf() {
      for (var e = nh, t = kg = nh = 0; t < e; ) {
        var a = cu[t];
        cu[t++] = null;
        var i = cu[t];
        cu[t++] = null;
        var o = cu[t];
        cu[t++] = null;
        var f = cu[t];
        if (cu[t++] = null, i !== null && o !== null) {
          var d = i.pending;
          d === null ? o.next = o : (o.next = d.next, d.next = o), i.pending = o;
        }
        f !== 0 && Gp(a, o, f);
      }
    }
    function Er(e, t, a, i) {
      cu[nh++] = e, cu[nh++] = t, cu[nh++] = a, cu[nh++] = i, kg |= i, e.lanes |= i, e = e.alternate, e !== null && (e.lanes |= i);
    }
    function Kh(e, t, a, i) {
      return Er(e, t, a, i), Ar(e);
    }
    function ia(e, t) {
      return Er(e, null, null, t), Ar(e);
    }
    function Gp(e, t, a) {
      e.lanes |= a;
      var i = e.alternate;
      i !== null && (i.lanes |= a);
      for (var o = !1, f = e.return; f !== null; )
        f.childLanes |= a, i = f.alternate, i !== null && (i.childLanes |= a), f.tag === 22 && (e = f.stateNode, e === null || e._visibility & Nv || (o = !0)), e = f, f = f.return;
      return e.tag === 3 ? (f = e.stateNode, o && t !== null && (o = 31 - Ql(a), e = f.hiddenUpdates, i = e[o], i === null ? e[o] = [t] : i.push(t), t.lane = a | 536870912), f) : null;
    }
    function Ar(e) {
      if (dp > FS)
        throw Ps = dp = 0, hp = R0 = null, Error(
          "Maximum update depth exceeded. This can happen when a component repeatedly calls setState inside componentWillUpdate or componentDidUpdate. React limits the number of nested updates to prevent infinite loops."
        );
      Ps > IS && (Ps = 0, hp = null, console.error(
        "Maximum update depth exceeded. This can happen when a component calls setState inside useEffect, but useEffect either doesn't have a dependency array, or one of the dependencies changes on every render."
      )), e.alternate === null && (e.flags & 4098) !== 0 && Sn(e);
      for (var t = e, a = t.return; a !== null; )
        t.alternate === null && (t.flags & 4098) !== 0 && Sn(e), t = a, a = t.return;
      return t.tag === 3 ? t.stateNode : null;
    }
    function ac(e) {
      if (ou === null) return e;
      var t = ou(e);
      return t === void 0 ? e : t.current;
    }
    function Jh(e) {
      if (ou === null) return e;
      var t = ou(e);
      return t === void 0 ? e != null && typeof e.render == "function" && (t = ac(e.render), e.render !== t) ? (t = { $$typeof: Yu, render: t }, e.displayName !== void 0 && (t.displayName = e.displayName), t) : e : t.current;
    }
    function Lp(e, t) {
      if (ou === null) return !1;
      var a = e.elementType;
      t = t.type;
      var i = !1, o = typeof t == "object" && t !== null ? t.$$typeof : null;
      switch (e.tag) {
        case 1:
          typeof t == "function" && (i = !0);
          break;
        case 0:
          (typeof t == "function" || o === Ca) && (i = !0);
          break;
        case 11:
          (o === Yu || o === Ca) && (i = !0);
          break;
        case 14:
        case 15:
          (o === Ms || o === Ca) && (i = !0);
          break;
        default:
          return !1;
      }
      return !!(i && (e = ou(a), e !== void 0 && e === ou(t)));
    }
    function Vp(e) {
      ou !== null && typeof WeakSet == "function" && (uh === null && (uh = /* @__PURE__ */ new WeakSet()), uh.add(e));
    }
    function Nf(e, t, a) {
      var i = e.alternate, o = e.child, f = e.sibling, d = e.tag, h = e.type, v = null;
      switch (d) {
        case 0:
        case 15:
        case 1:
          v = h;
          break;
        case 11:
          v = h.render;
      }
      if (ou === null)
        throw Error("Expected resolveFamily to be set during hot reload.");
      var b = !1;
      h = !1, v !== null && (v = ou(v), v !== void 0 && (a.has(v) ? h = !0 : t.has(v) && (d === 1 ? h = !0 : b = !0))), uh !== null && (uh.has(e) || i !== null && uh.has(i)) && (h = !0), h && (e._debugNeedsRemount = !0), (h || b) && (i = ia(e, 2), i !== null && Kt(i, e, 2)), o === null || h || Nf(
        o,
        t,
        a
      ), f !== null && Nf(
        f,
        t,
        a
      );
    }
    function wf(e, t, a, i) {
      this.tag = e, this.key = a, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.refCleanup = this.ref = null, this.pendingProps = t, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = i, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null, this.actualDuration = -0, this.actualStartTime = -1.1, this.treeBaseDuration = this.selfBaseDuration = -0, this._debugTask = this._debugStack = this._debugOwner = this._debugInfo = null, this._debugNeedsRemount = !1, this._debugHookTypes = null, n1 || typeof Object.preventExtensions != "function" || Object.preventExtensions(this);
    }
    function kh(e) {
      return e = e.prototype, !(!e || !e.isReactComponent);
    }
    function Cn(e, t) {
      var a = e.alternate;
      switch (a === null ? (a = z(
        e.tag,
        t,
        e.key,
        e.mode
      ), a.elementType = e.elementType, a.type = e.type, a.stateNode = e.stateNode, a._debugOwner = e._debugOwner, a._debugStack = e._debugStack, a._debugTask = e._debugTask, a._debugHookTypes = e._debugHookTypes, a.alternate = e, e.alternate = a) : (a.pendingProps = t, a.type = e.type, a.flags = 0, a.subtreeFlags = 0, a.deletions = null, a.actualDuration = -0, a.actualStartTime = -1.1), a.flags = e.flags & 65011712, a.childLanes = e.childLanes, a.lanes = e.lanes, a.child = e.child, a.memoizedProps = e.memoizedProps, a.memoizedState = e.memoizedState, a.updateQueue = e.updateQueue, t = e.dependencies, a.dependencies = t === null ? null : {
        lanes: t.lanes,
        firstContext: t.firstContext,
        _debugThenableState: t._debugThenableState
      }, a.sibling = e.sibling, a.index = e.index, a.ref = e.ref, a.refCleanup = e.refCleanup, a.selfBaseDuration = e.selfBaseDuration, a.treeBaseDuration = e.treeBaseDuration, a._debugInfo = e._debugInfo, a._debugNeedsRemount = e._debugNeedsRemount, a.tag) {
        case 0:
        case 15:
          a.type = ac(e.type);
          break;
        case 1:
          a.type = ac(e.type);
          break;
        case 11:
          a.type = Jh(e.type);
      }
      return a;
    }
    function $h(e, t) {
      e.flags &= 65011714;
      var a = e.alternate;
      return a === null ? (e.childLanes = 0, e.lanes = t, e.child = null, e.subtreeFlags = 0, e.memoizedProps = null, e.memoizedState = null, e.updateQueue = null, e.dependencies = null, e.stateNode = null, e.selfBaseDuration = 0, e.treeBaseDuration = 0) : (e.childLanes = a.childLanes, e.lanes = a.lanes, e.child = a.child, e.subtreeFlags = 0, e.deletions = null, e.memoizedProps = a.memoizedProps, e.memoizedState = a.memoizedState, e.updateQueue = a.updateQueue, e.type = a.type, t = a.dependencies, e.dependencies = t === null ? null : {
        lanes: t.lanes,
        firstContext: t.firstContext,
        _debugThenableState: t._debugThenableState
      }, e.selfBaseDuration = a.selfBaseDuration, e.treeBaseDuration = a.treeBaseDuration), e;
    }
    function Rr(e, t, a, i, o, f) {
      var d = 0, h = e;
      if (typeof e == "function")
        kh(e) && (d = 1), h = ac(h);
      else if (typeof e == "string")
        d = O(), d = Xo(e, a, d) ? 26 : e === "html" || e === "head" || e === "body" ? 27 : 5;
      else
        e: switch (e) {
          case Em:
            return t = z(31, a, t, o), t.elementType = Em, t.lanes = f, t;
          case Le:
            return ui(
              a.children,
              o,
              f,
              t
            );
          case Zo:
            d = 8, o |= Sa, o |= Ku;
            break;
          case Ko:
            return e = a, i = o, typeof e.id != "string" && console.error(
              'Profiler must specify an "id" of type `string` as a prop. Received the type `%s` instead.',
              typeof e.id
            ), t = z(12, e, t, i | ta), t.elementType = Ko, t.lanes = f, t.stateNode = { effectDuration: 0, passiveEffectDuration: 0 }, t;
          case Jo:
            return t = z(13, a, t, o), t.elementType = Jo, t.lanes = f, t;
          case Ci:
            return t = z(19, a, t, o), t.elementType = Ci, t.lanes = f, t;
          default:
            if (typeof e == "object" && e !== null)
              switch (e.$$typeof) {
                case Tm:
                case Ia:
                  d = 10;
                  break e;
                case Yd:
                  d = 9;
                  break e;
                case Yu:
                  d = 11, h = Jh(h);
                  break e;
                case Ms:
                  d = 14;
                  break e;
                case Ca:
                  d = 16, h = null;
                  break e;
              }
            h = "", (e === void 0 || typeof e == "object" && e !== null && Object.keys(e).length === 0) && (h += " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports."), e === null ? a = "null" : we(e) ? a = "array" : e !== void 0 && e.$$typeof === Ui ? (a = "<" + (De(e.type) || "Unknown") + " />", h = " Did you accidentally export a JSX literal instead of a component?") : a = typeof e, (d = i ? bt(i) : null) && (h += `

Check the render method of \`` + d + "`."), d = 29, a = Error(
              "Element type is invalid: expected a string (for built-in components) or a class/function (for composite components) but got: " + (a + "." + h)
            ), h = null;
        }
      return t = z(d, a, t, o), t.elementType = e, t.type = h, t.lanes = f, t._debugOwner = i, t;
    }
    function qf(e, t, a) {
      return t = Rr(
        e.type,
        e.key,
        e.props,
        e._owner,
        t,
        a
      ), t._debugOwner = e._owner, t._debugStack = e._debugStack, t._debugTask = e._debugTask, t;
    }
    function ui(e, t, a, i) {
      return e = z(7, e, i, t), e.lanes = a, e;
    }
    function ii(e, t, a) {
      return e = z(6, e, null, t), e.lanes = a, e;
    }
    function Wh(e, t, a) {
      return t = z(
        4,
        e.children !== null ? e.children : [],
        e.key,
        t
      ), t.lanes = a, t.stateNode = {
        containerInfo: e.containerInfo,
        pendingChildren: null,
        implementation: e.implementation
      }, t;
    }
    function nc(e, t) {
      fn(), ih[ch++] = qv, ih[ch++] = wv, wv = e, qv = t;
    }
    function Xp(e, t, a) {
      fn(), fu[su++] = Gc, fu[su++] = Lc, fu[su++] = Ls, Ls = e;
      var i = Gc;
      e = Lc;
      var o = 32 - Ql(i) - 1;
      i &= ~(1 << o), a += 1;
      var f = 32 - Ql(t) + o;
      if (30 < f) {
        var d = o - o % 5;
        f = (i & (1 << d) - 1).toString(32), i >>= d, o -= d, Gc = 1 << 32 - Ql(t) + o | a << o | i, Lc = f + e;
      } else
        Gc = 1 << f | a << o | i, Lc = e;
    }
    function Or(e) {
      fn(), e.return !== null && (nc(e, 1), Xp(e, 1, 0));
    }
    function Dr(e) {
      for (; e === wv; )
        wv = ih[--ch], ih[ch] = null, qv = ih[--ch], ih[ch] = null;
      for (; e === Ls; )
        Ls = fu[--su], fu[su] = null, Lc = fu[--su], fu[su] = null, Gc = fu[--su], fu[su] = null;
    }
    function fn() {
      mt || console.error(
        "Expected to be hydrating. This is a bug in React. Please file an issue."
      );
    }
    function sn(e, t) {
      if (e.return === null) {
        if (ru === null)
          ru = {
            fiber: e,
            children: [],
            serverProps: void 0,
            serverTail: [],
            distanceFromLeaf: t
          };
        else {
          if (ru.fiber !== e)
            throw Error(
              "Saw multiple hydration diff roots in a pass. This is a bug in React."
            );
          ru.distanceFromLeaf > t && (ru.distanceFromLeaf = t);
        }
        return ru;
      }
      var a = sn(
        e.return,
        t + 1
      ).children;
      return 0 < a.length && a[a.length - 1].fiber === e ? (a = a[a.length - 1], a.distanceFromLeaf > t && (a.distanceFromLeaf = t), a) : (t = {
        fiber: e,
        children: [],
        serverProps: void 0,
        serverTail: [],
        distanceFromLeaf: t
      }, a.push(t), t);
    }
    function Fh(e, t) {
      Vc || (e = sn(e, 0), e.serverProps = null, t !== null && (t = _d(t), e.serverTail.push(t)));
    }
    function xn(e) {
      var t = "", a = ru;
      throw a !== null && (ru = null, t = zf(a)), so(
        Ra(
          Error(
            `Hydration failed because the server rendered HTML didn't match the client. As a result this tree will be regenerated on the client. This can happen if a SSR-ed Client Component used:

- A server/client branch \`if (typeof window !== 'undefined')\`.
- Variable input such as \`Date.now()\` or \`Math.random()\` which changes each time it's called.
- Date formatting in a user's locale which doesn't match the server.
- External changing data without sending a snapshot of it along with the HTML.
- Invalid HTML tag nesting.

It can also happen if the client has a browser extension installed which messes with the HTML before React loaded.

https://react.dev/link/hydration-mismatch` + t
          ),
          e
        )
      ), $g;
    }
    function Ih(e) {
      var t = e.stateNode, a = e.type, i = e.memoizedProps;
      switch (t[Zl] = e, t[ga] = i, In(a, i), a) {
        case "dialog":
          Pe("cancel", t), Pe("close", t);
          break;
        case "iframe":
        case "object":
        case "embed":
          Pe("load", t);
          break;
        case "video":
        case "audio":
          for (a = 0; a < yp.length; a++)
            Pe(yp[a], t);
          break;
        case "source":
          Pe("error", t);
          break;
        case "img":
        case "image":
        case "link":
          Pe("error", t), Pe("load", t);
          break;
        case "details":
          Pe("toggle", t);
          break;
        case "input":
          ve("input", i), Pe("invalid", t), ei(t, i), Mp(
            t,
            i.value,
            i.defaultValue,
            i.checked,
            i.defaultChecked,
            i.type,
            i.name,
            !0
          ), bu(t);
          break;
        case "option":
          Hh(t, i);
          break;
        case "select":
          ve("select", i), Pe("invalid", t), Of(t, i);
          break;
        case "textarea":
          ve("textarea", i), Pe("invalid", t), _n(t, i), Nh(
            t,
            i.value,
            i.defaultValue,
            i.children
          ), bu(t);
      }
      a = i.children, typeof a != "string" && typeof a != "number" && typeof a != "bigint" || t.textContent === "" + a || i.suppressHydrationWarning === !0 || em(t.textContent, a) ? (i.popover != null && (Pe("beforetoggle", t), Pe("toggle", t)), i.onScroll != null && Pe("scroll", t), i.onScrollEnd != null && Pe("scrollend", t), i.onClick != null && (t.onclick = wu), t = !0) : t = !1, t || xn(e);
    }
    function Ph(e) {
      for (Na = e.return; Na; )
        switch (Na.tag) {
          case 5:
          case 13:
            Yi = !1;
            return;
          case 27:
          case 3:
            Yi = !0;
            return;
          default:
            Na = Na.return;
        }
    }
    function uc(e) {
      if (e !== Na) return !1;
      if (!mt)
        return Ph(e), mt = !0, !1;
      var t = e.tag, a;
      if ((a = t !== 3 && t !== 27) && ((a = t === 5) && (a = e.type, a = !(a !== "form" && a !== "button") || Pn(e.type, e.memoizedProps)), a = !a), a && nl) {
        for (a = nl; a; ) {
          var i = sn(e, 0), o = _d(a);
          i.serverTail.push(o), a = o.type === "Suspense" ? om(a) : Nl(a.nextSibling);
        }
        xn(e);
      }
      if (Ph(e), t === 13) {
        if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e)
          throw Error(
            "Expected to have a hydrated suspense instance. This error is likely caused by a bug in React. Please file an issue."
          );
        nl = om(e);
      } else
        t === 27 ? (t = nl, eu(e.type) ? (e = q0, q0 = null, nl = e) : nl = t) : nl = Na ? Nl(e.stateNode.nextSibling) : null;
      return !0;
    }
    function ic() {
      nl = Na = null, Vc = mt = !1;
    }
    function ey() {
      var e = Vs;
      return e !== null && (ja === null ? ja = e : ja.push.apply(
        ja,
        e
      ), Vs = null), e;
    }
    function so(e) {
      Vs === null ? Vs = [e] : Vs.push(e);
    }
    function ty() {
      var e = ru;
      if (e !== null) {
        ru = null;
        for (var t = zf(e); 0 < e.children.length; )
          e = e.children[0];
        ye(e.fiber, function() {
          console.error(
            `A tree hydrated but some attributes of the server rendered HTML didn't match the client properties. This won't be patched up. This can happen if a SSR-ed Client Component used:

- A server/client branch \`if (typeof window !== 'undefined')\`.
- Variable input such as \`Date.now()\` or \`Math.random()\` which changes each time it's called.
- Date formatting in a user's locale which doesn't match the server.
- External changing data without sending a snapshot of it along with the HTML.
- Invalid HTML tag nesting.

It can also happen if the client has a browser extension installed which messes with the HTML before React loaded.

%s%s`,
            "https://react.dev/link/hydration-mismatch",
            t
          );
        });
      }
    }
    function zr() {
      oh = jv = null, fh = !1;
    }
    function ci(e, t, a) {
      ze(Wg, t._currentValue, e), t._currentValue = a, ze(Fg, t._currentRenderer, e), t._currentRenderer !== void 0 && t._currentRenderer !== null && t._currentRenderer !== o1 && console.error(
        "Detected multiple renderers concurrently rendering the same context provider. This is currently unsupported."
      ), t._currentRenderer = o1;
    }
    function Ru(e, t) {
      e._currentValue = Wg.current;
      var a = Fg.current;
      Se(Fg, t), e._currentRenderer = a, Se(Wg, t);
    }
    function ly(e, t, a) {
      for (; e !== null; ) {
        var i = e.alternate;
        if ((e.childLanes & t) !== t ? (e.childLanes |= t, i !== null && (i.childLanes |= t)) : i !== null && (i.childLanes & t) !== t && (i.childLanes |= t), e === a) break;
        e = e.return;
      }
      e !== a && console.error(
        "Expected to find the propagation root when scheduling context work. This error is likely caused by a bug in React. Please file an issue."
      );
    }
    function ay(e, t, a, i) {
      var o = e.child;
      for (o !== null && (o.return = e); o !== null; ) {
        var f = o.dependencies;
        if (f !== null) {
          var d = o.child;
          f = f.firstContext;
          e: for (; f !== null; ) {
            var h = f;
            f = o;
            for (var v = 0; v < t.length; v++)
              if (h.context === t[v]) {
                f.lanes |= a, h = f.alternate, h !== null && (h.lanes |= a), ly(
                  f.return,
                  a,
                  e
                ), i || (d = null);
                break e;
              }
            f = h.next;
          }
        } else if (o.tag === 18) {
          if (d = o.return, d === null)
            throw Error(
              "We just came from a parent so we must have had a parent. This is a bug in React."
            );
          d.lanes |= a, f = d.alternate, f !== null && (f.lanes |= a), ly(
            d,
            a,
            e
          ), d = null;
        } else d = o.child;
        if (d !== null) d.return = o;
        else
          for (d = o; d !== null; ) {
            if (d === e) {
              d = null;
              break;
            }
            if (o = d.sibling, o !== null) {
              o.return = d.return, d = o;
              break;
            }
            d = d.return;
          }
        o = d;
      }
    }
    function Ul(e, t, a, i) {
      e = null;
      for (var o = t, f = !1; o !== null; ) {
        if (!f) {
          if ((o.flags & 524288) !== 0) f = !0;
          else if ((o.flags & 262144) !== 0) break;
        }
        if (o.tag === 10) {
          var d = o.alternate;
          if (d === null)
            throw Error("Should have a current fiber. This is a bug in React.");
          if (d = d.memoizedProps, d !== null) {
            var h = o.type;
            Ha(o.pendingProps.value, d.value) || (e !== null ? e.push(h) : e = [h]);
          }
        } else if (o === $o.current) {
          if (d = o.alternate, d === null)
            throw Error("Should have a current fiber. This is a bug in React.");
          d.memoizedState.memoizedState !== o.memoizedState.memoizedState && (e !== null ? e.push(gp) : e = [gp]);
        }
        o = o.return;
      }
      e !== null && ay(
        t,
        e,
        a,
        i
      ), t.flags |= 262144;
    }
    function oi(e) {
      for (e = e.firstContext; e !== null; ) {
        if (!Ha(
          e.context._currentValue,
          e.memoizedValue
        ))
          return !0;
        e = e.next;
      }
      return !1;
    }
    function fi(e) {
      jv = e, oh = null, e = e.dependencies, e !== null && (e.firstContext = null);
    }
    function Ct(e) {
      return fh && console.error(
        "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
      ), ny(jv, e);
    }
    function jf(e, t) {
      return jv === null && fi(e), ny(e, t);
    }
    function ny(e, t) {
      var a = t._currentValue;
      if (t = { context: t, memoizedValue: a, next: null }, oh === null) {
        if (e === null)
          throw Error(
            "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
          );
        oh = t, e.dependencies = {
          lanes: 0,
          firstContext: t,
          _debugThenableState: null
        }, e.flags |= 524288;
      } else oh = oh.next = t;
      return a;
    }
    function Bf() {
      return {
        controller: new qS(),
        data: /* @__PURE__ */ new Map(),
        refCount: 0
      };
    }
    function cc(e) {
      e.controller.signal.aborted && console.warn(
        "A cache instance was retained after it was already freed. This likely indicates a bug in React."
      ), e.refCount++;
    }
    function Hn(e) {
      e.refCount--, 0 > e.refCount && console.warn(
        "A cache instance was released after it was already freed. This likely indicates a bug in React."
      ), e.refCount === 0 && jS(BS, function() {
        e.controller.abort();
      });
    }
    function rn() {
      var e = Xs;
      return Xs = 0, e;
    }
    function si(e) {
      var t = Xs;
      return Xs = e, t;
    }
    function oc(e) {
      var t = Xs;
      return Xs += e, t;
    }
    function Mr(e) {
      tn = sh(), 0 > e.actualStartTime && (e.actualStartTime = tn);
    }
    function Ou(e) {
      if (0 <= tn) {
        var t = sh() - tn;
        e.actualDuration += t, e.selfBaseDuration = t, tn = -1;
      }
    }
    function fc(e) {
      if (0 <= tn) {
        var t = sh() - tn;
        e.actualDuration += t, tn = -1;
      }
    }
    function Ga() {
      if (0 <= tn) {
        var e = sh() - tn;
        tn = -1, Xs += e;
      }
    }
    function dn() {
      tn = sh();
    }
    function Nn(e) {
      for (var t = e.child; t; )
        e.actualDuration += t.actualDuration, t = t.sibling;
    }
    function Qp(e, t) {
      if (Zm === null) {
        var a = Zm = [];
        Ig = 0, Qs = $y(), rh = {
          status: "pending",
          value: void 0,
          then: function(i) {
            a.push(i);
          }
        };
      }
      return Ig++, t.then(uy, uy), t;
    }
    function uy() {
      if (--Ig === 0 && Zm !== null) {
        rh !== null && (rh.status = "fulfilled");
        var e = Zm;
        Zm = null, Qs = 0, rh = null;
        for (var t = 0; t < e.length; t++) (0, e[t])();
      }
    }
    function Zp(e, t) {
      var a = [], i = {
        status: "pending",
        value: null,
        reason: null,
        then: function(o) {
          a.push(o);
        }
      };
      return e.then(
        function() {
          i.status = "fulfilled", i.value = t;
          for (var o = 0; o < a.length; o++) (0, a[o])(t);
        },
        function(o) {
          for (i.status = "rejected", i.reason = o, o = 0; o < a.length; o++)
            (0, a[o])(void 0);
        }
      ), i;
    }
    function iy() {
      var e = Zs.current;
      return e !== null ? e : xt.pooledCache;
    }
    function _r(e, t) {
      t === null ? ze(Zs, Zs.current, e) : ze(Zs, t.pool, e);
    }
    function Kp() {
      var e = iy();
      return e === null ? null : { parent: jl._currentValue, pool: e };
    }
    function cy() {
      return { didWarnAboutUncachedPromise: !1, thenables: [] };
    }
    function oy(e) {
      return e = e.status, e === "fulfilled" || e === "rejected";
    }
    function ro() {
    }
    function La(e, t, a) {
      L.actQueue !== null && (L.didUsePromise = !0);
      var i = e.thenables;
      switch (a = i[a], a === void 0 ? i.push(t) : a !== t && (e.didWarnAboutUncachedPromise || (e.didWarnAboutUncachedPromise = !0, console.error(
        "A component was suspended by an uncached promise. Creating promises inside a Client Component or hook is not yet supported, except via a Suspense-compatible library or framework."
      )), t.then(ro, ro), t = a), t.status) {
        case "fulfilled":
          return t.value;
        case "rejected":
          throw e = t.reason, Oa(e), e;
        default:
          if (typeof t.status == "string")
            t.then(ro, ro);
          else {
            if (e = xt, e !== null && 100 < e.shellSuspendCounter)
              throw Error(
                "An unknown Component is an async Client Component. Only Server Components can be async at the moment. This error is often caused by accidentally adding `'use client'` to a module that was originally written for the server."
              );
            e = t, e.status = "pending", e.then(
              function(o) {
                if (t.status === "pending") {
                  var f = t;
                  f.status = "fulfilled", f.value = o;
                }
              },
              function(o) {
                if (t.status === "pending") {
                  var f = t;
                  f.status = "rejected", f.reason = o;
                }
              }
            );
          }
          switch (t.status) {
            case "fulfilled":
              return t.value;
            case "rejected":
              throw e = t.reason, Oa(e), e;
          }
          throw Pm = t, Xv = !0, Im;
      }
    }
    function fy() {
      if (Pm === null)
        throw Error(
          "Expected a suspended thenable. This is a bug in React. Please file an issue."
        );
      var e = Pm;
      return Pm = null, Xv = !1, e;
    }
    function Oa(e) {
      if (e === Im || e === Vv)
        throw Error(
          "Hooks are not supported inside an async component. This error is often caused by accidentally adding `'use client'` to a module that was originally written for the server."
        );
    }
    function ca(e) {
      e.updateQueue = {
        baseState: e.memoizedState,
        firstBaseUpdate: null,
        lastBaseUpdate: null,
        shared: { pending: null, lanes: 0, hiddenCallbacks: null },
        callbacks: null
      };
    }
    function ri(e, t) {
      e = e.updateQueue, t.updateQueue === e && (t.updateQueue = {
        baseState: e.baseState,
        firstBaseUpdate: e.firstBaseUpdate,
        lastBaseUpdate: e.lastBaseUpdate,
        shared: e.shared,
        callbacks: null
      });
    }
    function wn(e) {
      return {
        lane: e,
        tag: h1,
        payload: null,
        callback: null,
        next: null
      };
    }
    function hn(e, t, a) {
      var i = e.updateQueue;
      if (i === null) return null;
      if (i = i.shared, t0 === i && !p1) {
        var o = re(e);
        console.error(
          `An update (setState, replaceState, or forceUpdate) was scheduled from inside an update function. Update functions should be pure, with zero side-effects. Consider using componentDidUpdate or a callback.

Please update the following component: %s`,
          o
        ), p1 = !0;
      }
      return (Et & qa) !== An ? (o = i.pending, o === null ? t.next = t : (t.next = o.next, o.next = t), i.pending = t, t = Ar(e), Gp(e, null, a), t) : (Er(e, i, t, a), Ar(e));
    }
    function di(e, t, a) {
      if (t = t.updateQueue, t !== null && (t = t.shared, (a & 4194048) !== 0)) {
        var i = t.lanes;
        i &= e.pendingLanes, a |= i, t.lanes = a, Pu(e, a);
      }
    }
    function ho(e, t) {
      var a = e.updateQueue, i = e.alternate;
      if (i !== null && (i = i.updateQueue, a === i)) {
        var o = null, f = null;
        if (a = a.firstBaseUpdate, a !== null) {
          do {
            var d = {
              lane: a.lane,
              tag: a.tag,
              payload: a.payload,
              callback: null,
              next: null
            };
            f === null ? o = f = d : f = f.next = d, a = a.next;
          } while (a !== null);
          f === null ? o = f = t : f = f.next = t;
        } else o = f = t;
        a = {
          baseState: i.baseState,
          firstBaseUpdate: o,
          lastBaseUpdate: f,
          shared: i.shared,
          callbacks: i.callbacks
        }, e.updateQueue = a;
        return;
      }
      e = a.lastBaseUpdate, e === null ? a.firstBaseUpdate = t : e.next = t, a.lastBaseUpdate = t;
    }
    function qn() {
      if (l0) {
        var e = rh;
        if (e !== null) throw e;
      }
    }
    function yo(e, t, a, i) {
      l0 = !1;
      var o = e.updateQueue;
      uf = !1, t0 = o.shared;
      var f = o.firstBaseUpdate, d = o.lastBaseUpdate, h = o.shared.pending;
      if (h !== null) {
        o.shared.pending = null;
        var v = h, b = v.next;
        v.next = null, d === null ? f = b : d.next = b, d = v;
        var B = e.alternate;
        B !== null && (B = B.updateQueue, h = B.lastBaseUpdate, h !== d && (h === null ? B.firstBaseUpdate = b : h.next = b, B.lastBaseUpdate = v));
      }
      if (f !== null) {
        var X = o.baseState;
        d = 0, B = b = v = null, h = f;
        do {
          var N = h.lane & -536870913, Q = N !== h.lane;
          if (Q ? (ut & N) === N : (i & N) === N) {
            N !== 0 && N === Qs && (l0 = !0), B !== null && (B = B.next = {
              lane: 0,
              tag: h.tag,
              payload: h.payload,
              callback: null,
              next: null
            });
            e: {
              N = e;
              var me = h, xe = t, Ht = a;
              switch (me.tag) {
                case y1:
                  if (me = me.payload, typeof me == "function") {
                    fh = !0;
                    var ft = me.call(
                      Ht,
                      X,
                      xe
                    );
                    if (N.mode & Sa) {
                      oe(!0);
                      try {
                        me.call(Ht, X, xe);
                      } finally {
                        oe(!1);
                      }
                    }
                    fh = !1, X = ft;
                    break e;
                  }
                  X = me;
                  break e;
                case e0:
                  N.flags = N.flags & -65537 | 128;
                case h1:
                  if (ft = me.payload, typeof ft == "function") {
                    if (fh = !0, me = ft.call(
                      Ht,
                      X,
                      xe
                    ), N.mode & Sa) {
                      oe(!0);
                      try {
                        ft.call(Ht, X, xe);
                      } finally {
                        oe(!1);
                      }
                    }
                    fh = !1;
                  } else me = ft;
                  if (me == null) break e;
                  X = Je({}, X, me);
                  break e;
                case m1:
                  uf = !0;
              }
            }
            N = h.callback, N !== null && (e.flags |= 64, Q && (e.flags |= 8192), Q = o.callbacks, Q === null ? o.callbacks = [N] : Q.push(N));
          } else
            Q = {
              lane: N,
              tag: h.tag,
              payload: h.payload,
              callback: h.callback,
              next: null
            }, B === null ? (b = B = Q, v = X) : B = B.next = Q, d |= N;
          if (h = h.next, h === null) {
            if (h = o.shared.pending, h === null)
              break;
            Q = h, h = Q.next, Q.next = null, o.lastBaseUpdate = Q, o.shared.pending = null;
          }
        } while (!0);
        B === null && (v = X), o.baseState = v, o.firstBaseUpdate = b, o.lastBaseUpdate = B, f === null && (o.shared.lanes = 0), sf |= d, e.lanes = d, e.memoizedState = X;
      }
      t0 = null;
    }
    function Yf(e, t) {
      if (typeof e != "function")
        throw Error(
          "Invalid argument passed as callback. Expected a function. Instead received: " + e
        );
      e.call(t);
    }
    function mo(e, t) {
      var a = e.shared.hiddenCallbacks;
      if (a !== null)
        for (e.shared.hiddenCallbacks = null, e = 0; e < a.length; e++)
          Yf(a[e], t);
    }
    function Jp(e, t) {
      var a = e.callbacks;
      if (a !== null)
        for (e.callbacks = null, e = 0; e < a.length; e++)
          Yf(a[e], t);
    }
    function oa(e, t) {
      var a = Vi;
      ze(Qv, a, e), ze(dh, t, e), Vi = a | t.baseLanes;
    }
    function Gf(e) {
      ze(Qv, Vi, e), ze(
        dh,
        dh.current,
        e
      );
    }
    function yn(e) {
      Vi = Qv.current, Se(dh, e), Se(Qv, e);
    }
    function $e() {
      var e = V;
      yu === null ? yu = [e] : yu.push(e);
    }
    function ee() {
      var e = V;
      if (yu !== null && (Qc++, yu[Qc] !== e)) {
        var t = re(qe);
        if (!v1.has(t) && (v1.add(t), yu !== null)) {
          for (var a = "", i = 0; i <= Qc; i++) {
            var o = yu[i], f = i === Qc ? e : o;
            for (o = i + 1 + ". " + o; 30 > o.length; )
              o += " ";
            o += f + `
`, a += o;
          }
          console.error(
            `React has detected a change in the order of Hooks called by %s. This will lead to bugs and errors if not fixed. For more information, read the Rules of Hooks: https://react.dev/link/rules-of-hooks

   Previous render            Next render
   ------------------------------------------------------
%s   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`,
            t,
            a
          );
        }
      }
    }
    function Va(e) {
      e == null || we(e) || console.error(
        "%s received a final argument that is not an array (instead, received `%s`). When specified, the final argument must be an array.",
        V,
        typeof e
      );
    }
    function po() {
      var e = re(qe);
      b1.has(e) || (b1.add(e), console.error(
        "ReactDOM.useFormState has been renamed to React.useActionState. Please update %s to use React.useActionState.",
        e
      ));
    }
    function Lt() {
      throw Error(
        `Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:
1. You might have mismatching versions of React and the renderer (such as React DOM)
2. You might be breaking the Rules of Hooks
3. You might have more than one copy of React in the same app
See https://react.dev/link/invalid-hook-call for tips about how to debug and fix this problem.`
      );
    }
    function hi(e, t) {
      if (tp) return !1;
      if (t === null)
        return console.error(
          "%s received a final argument during this render, but not during the previous render. Even though the final argument is optional, its type cannot change between renders.",
          V
        ), !1;
      e.length !== t.length && console.error(
        `The final argument passed to %s changed size between renders. The order and size of this array must remain constant.

Previous: %s
Incoming: %s`,
        V,
        "[" + t.join(", ") + "]",
        "[" + e.join(", ") + "]"
      );
      for (var a = 0; a < t.length && a < e.length; a++)
        if (!Ha(e[a], t[a])) return !1;
      return !0;
    }
    function yi(e, t, a, i, o, f) {
      cf = f, qe = t, yu = e !== null ? e._debugHookTypes : null, Qc = -1, tp = e !== null && e.type !== t.type, (Object.prototype.toString.call(a) === "[object AsyncFunction]" || Object.prototype.toString.call(a) === "[object AsyncGeneratorFunction]") && (f = re(qe), a0.has(f) || (a0.add(f), console.error(
        "%s is an async Client Component. Only Server Components can be async at the moment. This error is often caused by accidentally adding `'use client'` to a module that was originally written for the server.",
        f === null ? "An unknown Component" : "<" + f + ">"
      ))), t.memoizedState = null, t.updateQueue = null, t.lanes = 0, L.H = e !== null && e.memoizedState !== null ? u0 : yu !== null ? S1 : n0, Js = f = (t.mode & Sa) !== Bt;
      var d = i0(a, i, o);
      if (Js = !1, yh && (d = vo(
        t,
        a,
        i,
        o
      )), f) {
        oe(!0);
        try {
          d = vo(
            t,
            a,
            i,
            o
          );
        } finally {
          oe(!1);
        }
      }
      return Lf(e, t), d;
    }
    function Lf(e, t) {
      t._debugHookTypes = yu, t.dependencies === null ? Xc !== null && (t.dependencies = {
        lanes: 0,
        firstContext: null,
        _debugThenableState: Xc
      }) : t.dependencies._debugThenableState = Xc, L.H = Jv;
      var a = Ut !== null && Ut.next !== null;
      if (cf = 0, yu = V = Al = Ut = qe = null, Qc = -1, e !== null && (e.flags & 65011712) !== (t.flags & 65011712) && console.error(
        "Internal React error: Expected static flag was missing. Please notify the React team."
      ), Zv = !1, ep = 0, Xc = null, a)
        throw Error(
          "Rendered fewer hooks than expected. This may be caused by an accidental early return statement."
        );
      e === null || Kl || (e = e.dependencies, e !== null && oi(e) && (Kl = !0)), Xv ? (Xv = !1, e = !0) : e = !1, e && (t = re(t) || "Unknown", g1.has(t) || a0.has(t) || (g1.add(t), console.error(
        "`use` was called from inside a try/catch block. This is not allowed and can lead to unexpected behavior. To handle errors triggered by `use`, wrap your component in a error boundary."
      )));
    }
    function vo(e, t, a, i) {
      qe = e;
      var o = 0;
      do {
        if (yh && (Xc = null), ep = 0, yh = !1, o >= GS)
          throw Error(
            "Too many re-renders. React limits the number of renders to prevent an infinite loop."
          );
        if (o += 1, tp = !1, Al = Ut = null, e.updateQueue != null) {
          var f = e.updateQueue;
          f.lastEffect = null, f.events = null, f.stores = null, f.memoCache != null && (f.memoCache.index = 0);
        }
        Qc = -1, L.H = T1, f = i0(t, a, i);
      } while (yh);
      return f;
    }
    function Xa() {
      var e = L.H, t = e.useState()[0];
      return t = typeof t.then == "function" ? sc(t) : t, e = e.useState()[0], (Ut !== null ? Ut.memoizedState : null) !== e && (qe.flags |= 1024), t;
    }
    function fa() {
      var e = Kv !== 0;
      return Kv = 0, e;
    }
    function Du(e, t, a) {
      t.updateQueue = e.updateQueue, t.flags = (t.mode & Ku) !== Bt ? t.flags & -402655237 : t.flags & -2053, e.lanes &= ~a;
    }
    function mn(e) {
      if (Zv) {
        for (e = e.memoizedState; e !== null; ) {
          var t = e.queue;
          t !== null && (t.pending = null), e = e.next;
        }
        Zv = !1;
      }
      cf = 0, yu = Al = Ut = qe = null, Qc = -1, V = null, yh = !1, ep = Kv = 0, Xc = null;
    }
    function Qt() {
      var e = {
        memoizedState: null,
        baseState: null,
        baseQueue: null,
        queue: null,
        next: null
      };
      return Al === null ? qe.memoizedState = Al = e : Al = Al.next = e, Al;
    }
    function ot() {
      if (Ut === null) {
        var e = qe.alternate;
        e = e !== null ? e.memoizedState : null;
      } else e = Ut.next;
      var t = Al === null ? qe.memoizedState : Al.next;
      if (t !== null)
        Al = t, Ut = e;
      else {
        if (e === null)
          throw qe.alternate === null ? Error(
            "Update hook called on initial render. This is likely a bug in React. Please file an issue."
          ) : Error("Rendered more hooks than during the previous render.");
        Ut = e, e = {
          memoizedState: Ut.memoizedState,
          baseState: Ut.baseState,
          baseQueue: Ut.baseQueue,
          queue: Ut.queue,
          next: null
        }, Al === null ? qe.memoizedState = Al = e : Al = Al.next = e;
      }
      return Al;
    }
    function Ur() {
      return { lastEffect: null, events: null, stores: null, memoCache: null };
    }
    function sc(e) {
      var t = ep;
      return ep += 1, Xc === null && (Xc = cy()), e = La(Xc, e, t), t = qe, (Al === null ? t.memoizedState : Al.next) === null && (t = t.alternate, L.H = t !== null && t.memoizedState !== null ? u0 : n0), e;
    }
    function jn(e) {
      if (e !== null && typeof e == "object") {
        if (typeof e.then == "function") return sc(e);
        if (e.$$typeof === Ia) return Ct(e);
      }
      throw Error("An unsupported type was passed to use(): " + String(e));
    }
    function Pt(e) {
      var t = null, a = qe.updateQueue;
      if (a !== null && (t = a.memoCache), t == null) {
        var i = qe.alternate;
        i !== null && (i = i.updateQueue, i !== null && (i = i.memoCache, i != null && (t = {
          data: i.data.map(function(o) {
            return o.slice();
          }),
          index: 0
        })));
      }
      if (t == null && (t = { data: [], index: 0 }), a === null && (a = Ur(), qe.updateQueue = a), a.memoCache = t, a = t.data[t.index], a === void 0 || tp)
        for (a = t.data[t.index] = Array(e), i = 0; i < e; i++)
          a[i] = Ev;
      else
        a.length !== e && console.error(
          "Expected a constant size argument for each invocation of useMemoCache. The previous cache was allocated with size %s but size %s was requested.",
          a.length,
          e
        );
      return t.index++, a;
    }
    function dt(e, t) {
      return typeof t == "function" ? t(e) : t;
    }
    function rt(e, t, a) {
      var i = Qt();
      if (a !== void 0) {
        var o = a(t);
        if (Js) {
          oe(!0);
          try {
            a(t);
          } finally {
            oe(!1);
          }
        }
      } else o = t;
      return i.memoizedState = i.baseState = o, e = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: e,
        lastRenderedState: o
      }, i.queue = e, e = e.dispatch = gy.bind(
        null,
        qe,
        e
      ), [i.memoizedState, e];
    }
    function Qa(e) {
      var t = ot();
      return Za(t, Ut, e);
    }
    function Za(e, t, a) {
      var i = e.queue;
      if (i === null)
        throw Error(
          "Should have a queue. You are likely calling Hooks conditionally, which is not allowed. (https://react.dev/link/invalid-hook-call)"
        );
      i.lastRenderedReducer = a;
      var o = e.baseQueue, f = i.pending;
      if (f !== null) {
        if (o !== null) {
          var d = o.next;
          o.next = f.next, f.next = d;
        }
        t.baseQueue !== o && console.error(
          "Internal error: Expected work-in-progress queue to be a clone. This is a bug in React."
        ), t.baseQueue = o = f, i.pending = null;
      }
      if (f = e.baseState, o === null) e.memoizedState = f;
      else {
        t = o.next;
        var h = d = null, v = null, b = t, B = !1;
        do {
          var X = b.lane & -536870913;
          if (X !== b.lane ? (ut & X) === X : (cf & X) === X) {
            var N = b.revertLane;
            if (N === 0)
              v !== null && (v = v.next = {
                lane: 0,
                revertLane: 0,
                action: b.action,
                hasEagerState: b.hasEagerState,
                eagerState: b.eagerState,
                next: null
              }), X === Qs && (B = !0);
            else if ((cf & N) === N) {
              b = b.next, N === Qs && (B = !0);
              continue;
            } else
              X = {
                lane: 0,
                revertLane: b.revertLane,
                action: b.action,
                hasEagerState: b.hasEagerState,
                eagerState: b.eagerState,
                next: null
              }, v === null ? (h = v = X, d = f) : v = v.next = X, qe.lanes |= N, sf |= N;
            X = b.action, Js && a(f, X), f = b.hasEagerState ? b.eagerState : a(f, X);
          } else
            N = {
              lane: X,
              revertLane: b.revertLane,
              action: b.action,
              hasEagerState: b.hasEagerState,
              eagerState: b.eagerState,
              next: null
            }, v === null ? (h = v = N, d = f) : v = v.next = N, qe.lanes |= X, sf |= X;
          b = b.next;
        } while (b !== null && b !== t);
        if (v === null ? d = f : v.next = h, !Ha(f, e.memoizedState) && (Kl = !0, B && (a = rh, a !== null)))
          throw a;
        e.memoizedState = f, e.baseState = d, e.baseQueue = v, i.lastRenderedState = f;
      }
      return o === null && (i.lanes = 0), [e.memoizedState, i.dispatch];
    }
    function rc(e) {
      var t = ot(), a = t.queue;
      if (a === null)
        throw Error(
          "Should have a queue. You are likely calling Hooks conditionally, which is not allowed. (https://react.dev/link/invalid-hook-call)"
        );
      a.lastRenderedReducer = e;
      var i = a.dispatch, o = a.pending, f = t.memoizedState;
      if (o !== null) {
        a.pending = null;
        var d = o = o.next;
        do
          f = e(f, d.action), d = d.next;
        while (d !== o);
        Ha(f, t.memoizedState) || (Kl = !0), t.memoizedState = f, t.baseQueue === null && (t.baseState = f), a.lastRenderedState = f;
      }
      return [f, i];
    }
    function zu(e, t, a) {
      var i = qe, o = Qt();
      if (mt) {
        if (a === void 0)
          throw Error(
            "Missing getServerSnapshot, which is required for server-rendered content. Will revert to client rendering."
          );
        var f = a();
        hh || f === a() || (console.error(
          "The result of getServerSnapshot should be cached to avoid an infinite loop"
        ), hh = !0);
      } else {
        if (f = t(), hh || (a = t(), Ha(f, a) || (console.error(
          "The result of getSnapshot should be cached to avoid an infinite loop"
        ), hh = !0)), xt === null)
          throw Error(
            "Expected a work-in-progress root. This is a bug in React. Please file an issue."
          );
        (ut & 124) !== 0 || sy(i, t, f);
      }
      return o.memoizedState = f, a = { value: f, getSnapshot: t }, o.queue = a, Hr(
        bo.bind(null, i, a, e),
        [e]
      ), i.flags |= 2048, Yn(
        hu | Bl,
        pi(),
        go.bind(
          null,
          i,
          a,
          f,
          t
        ),
        null
      ), f;
    }
    function Vf(e, t, a) {
      var i = qe, o = ot(), f = mt;
      if (f) {
        if (a === void 0)
          throw Error(
            "Missing getServerSnapshot, which is required for server-rendered content. Will revert to client rendering."
          );
        a = a();
      } else if (a = t(), !hh) {
        var d = t();
        Ha(a, d) || (console.error(
          "The result of getSnapshot should be cached to avoid an infinite loop"
        ), hh = !0);
      }
      (d = !Ha(
        (Ut || o).memoizedState,
        a
      )) && (o.memoizedState = a, Kl = !0), o = o.queue;
      var h = bo.bind(null, i, o, e);
      if (sl(2048, Bl, h, [e]), o.getSnapshot !== t || d || Al !== null && Al.memoizedState.tag & hu) {
        if (i.flags |= 2048, Yn(
          hu | Bl,
          pi(),
          go.bind(
            null,
            i,
            o,
            a,
            t
          ),
          null
        ), xt === null)
          throw Error(
            "Expected a work-in-progress root. This is a bug in React. Please file an issue."
          );
        f || (cf & 124) !== 0 || sy(i, t, a);
      }
      return a;
    }
    function sy(e, t, a) {
      e.flags |= 16384, e = { getSnapshot: t, value: a }, t = qe.updateQueue, t === null ? (t = Ur(), qe.updateQueue = t, t.stores = [e]) : (a = t.stores, a === null ? t.stores = [e] : a.push(e));
    }
    function go(e, t, a, i) {
      t.value = a, t.getSnapshot = i, ry(t) && So(e);
    }
    function bo(e, t, a) {
      return a(function() {
        ry(t) && So(e);
      });
    }
    function ry(e) {
      var t = e.getSnapshot;
      e = e.value;
      try {
        var a = t();
        return !Ha(e, a);
      } catch {
        return !0;
      }
    }
    function So(e) {
      var t = ia(e, 2);
      t !== null && Kt(t, e, 2);
    }
    function Xf(e) {
      var t = Qt();
      if (typeof e == "function") {
        var a = e;
        if (e = a(), Js) {
          oe(!0);
          try {
            a();
          } finally {
            oe(!1);
          }
        }
      }
      return t.memoizedState = t.baseState = e, t.queue = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: dt,
        lastRenderedState: e
      }, t;
    }
    function Mu(e) {
      e = Xf(e);
      var t = e.queue, a = Ro.bind(null, qe, t);
      return t.dispatch = a, [e.memoizedState, a];
    }
    function pn(e) {
      var t = Qt();
      t.memoizedState = t.baseState = e;
      var a = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: null,
        lastRenderedState: null
      };
      return t.queue = a, t = Vr.bind(
        null,
        qe,
        !0,
        a
      ), a.dispatch = t, [e, t];
    }
    function _u(e, t) {
      var a = ot();
      return Bn(a, Ut, e, t);
    }
    function Bn(e, t, a, i) {
      return e.baseState = a, Za(
        e,
        Ut,
        typeof i == "function" ? i : dt
      );
    }
    function Cr(e, t) {
      var a = ot();
      return Ut !== null ? Bn(a, Ut, e, t) : (a.baseState = e, [e, a.queue.dispatch]);
    }
    function dy(e, t, a, i, o) {
      if (Wf(e))
        throw Error("Cannot update form state while rendering.");
      if (e = t.action, e !== null) {
        var f = {
          payload: o,
          action: e,
          next: null,
          isTransition: !0,
          status: "pending",
          value: null,
          reason: null,
          listeners: [],
          then: function(d) {
            f.listeners.push(d);
          }
        };
        L.T !== null ? a(!0) : f.isTransition = !1, i(f), a = t.pending, a === null ? (f.next = t.pending = f, To(t, f)) : (f.next = a.next, t.pending = a.next = f);
      }
    }
    function To(e, t) {
      var a = t.action, i = t.payload, o = e.state;
      if (t.isTransition) {
        var f = L.T, d = {};
        L.T = d, L.T._updatedFibers = /* @__PURE__ */ new Set();
        try {
          var h = a(o, i), v = L.S;
          v !== null && v(d, h), Qf(e, t, h);
        } catch (b) {
          gl(e, t, b);
        } finally {
          L.T = f, f === null && d._updatedFibers && (e = d._updatedFibers.size, d._updatedFibers.clear(), 10 < e && console.warn(
            "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
          ));
        }
      } else
        try {
          d = a(o, i), Qf(e, t, d);
        } catch (b) {
          gl(e, t, b);
        }
    }
    function Qf(e, t, a) {
      a !== null && typeof a == "object" && typeof a.then == "function" ? (a.then(
        function(i) {
          mi(e, t, i);
        },
        function(i) {
          return gl(e, t, i);
        }
      ), t.isTransition || console.error(
        "An async function with useActionState was called outside of a transition. This is likely not what you intended (for example, isPending will not update correctly). Either call the returned function inside startTransition, or pass it to an `action` or `formAction` prop."
      )) : mi(e, t, a);
    }
    function mi(e, t, a) {
      t.status = "fulfilled", t.value = a, Zf(t), e.state = a, t = e.pending, t !== null && (a = t.next, a === t ? e.pending = null : (a = a.next, t.next = a, To(e, a)));
    }
    function gl(e, t, a) {
      var i = e.pending;
      if (e.pending = null, i !== null) {
        i = i.next;
        do
          t.status = "rejected", t.reason = a, Zf(t), t = t.next;
        while (t !== i);
      }
      e.action = null;
    }
    function Zf(e) {
      e = e.listeners;
      for (var t = 0; t < e.length; t++) (0, e[t])();
    }
    function hy(e, t) {
      return t;
    }
    function Eo(e, t) {
      if (mt) {
        var a = xt.formState;
        if (a !== null) {
          e: {
            var i = qe;
            if (mt) {
              if (nl) {
                t: {
                  for (var o = nl, f = Yi; o.nodeType !== 8; ) {
                    if (!f) {
                      o = null;
                      break t;
                    }
                    if (o = Nl(
                      o.nextSibling
                    ), o === null) {
                      o = null;
                      break t;
                    }
                  }
                  f = o.data, o = f === x0 || f === bb ? o : null;
                }
                if (o) {
                  nl = Nl(
                    o.nextSibling
                  ), i = o.data === x0;
                  break e;
                }
              }
              xn(i);
            }
            i = !1;
          }
          i && (t = a[0]);
        }
      }
      return a = Qt(), a.memoizedState = a.baseState = t, i = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: hy,
        lastRenderedState: t
      }, a.queue = i, a = Ro.bind(
        null,
        qe,
        i
      ), i.dispatch = a, i = Xf(!1), f = Vr.bind(
        null,
        qe,
        !1,
        i.queue
      ), i = Qt(), o = {
        state: t,
        dispatch: null,
        action: e,
        pending: null
      }, i.queue = o, a = dy.bind(
        null,
        qe,
        o,
        f,
        a
      ), o.dispatch = a, i.memoizedState = e, [t, a, !1];
    }
    function xr(e) {
      var t = ot();
      return kp(t, Ut, e);
    }
    function kp(e, t, a) {
      if (t = Za(
        e,
        t,
        hy
      )[0], e = Qa(dt)[0], typeof t == "object" && t !== null && typeof t.then == "function")
        try {
          var i = sc(t);
        } catch (d) {
          throw d === Im ? Vv : d;
        }
      else i = t;
      t = ot();
      var o = t.queue, f = o.dispatch;
      return a !== t.memoizedState && (qe.flags |= 2048, Yn(
        hu | Bl,
        pi(),
        fl.bind(null, o, a),
        null
      )), [i, f, e];
    }
    function fl(e, t) {
      e.action = t;
    }
    function Ao(e) {
      var t = ot(), a = Ut;
      if (a !== null)
        return kp(t, a, e);
      ot(), t = t.memoizedState, a = ot();
      var i = a.queue.dispatch;
      return a.memoizedState = e, [t, i, !1];
    }
    function Yn(e, t, a, i) {
      return e = {
        tag: e,
        create: a,
        deps: i,
        inst: t,
        next: null
      }, t = qe.updateQueue, t === null && (t = Ur(), qe.updateQueue = t), a = t.lastEffect, a === null ? t.lastEffect = e.next = e : (i = a.next, a.next = e, e.next = i, t.lastEffect = e), e;
    }
    function pi() {
      return { destroy: void 0, resource: void 0 };
    }
    function Kf(e) {
      var t = Qt();
      return e = { current: e }, t.memoizedState = e;
    }
    function Ka(e, t, a, i) {
      var o = Qt();
      i = i === void 0 ? null : i, qe.flags |= e, o.memoizedState = Yn(
        hu | t,
        pi(),
        a,
        i
      );
    }
    function sl(e, t, a, i) {
      var o = ot();
      i = i === void 0 ? null : i;
      var f = o.memoizedState.inst;
      Ut !== null && i !== null && hi(i, Ut.memoizedState.deps) ? o.memoizedState = Yn(t, f, a, i) : (qe.flags |= e, o.memoizedState = Yn(
        hu | t,
        f,
        a,
        i
      ));
    }
    function Hr(e, t) {
      (qe.mode & Ku) !== Bt && (qe.mode & a1) === Bt ? Ka(276826112, Bl, e, t) : Ka(8390656, Bl, e, t);
    }
    function Nr(e, t) {
      var a = 4194308;
      return (qe.mode & Ku) !== Bt && (a |= 134217728), Ka(a, la, e, t);
    }
    function $p(e, t) {
      if (typeof t == "function") {
        e = e();
        var a = t(e);
        return function() {
          typeof a == "function" ? a() : t(null);
        };
      }
      if (t != null)
        return t.hasOwnProperty("current") || console.error(
          "Expected useImperativeHandle() first argument to either be a ref callback or React.createRef() object. Instead received: %s.",
          "an object with keys {" + Object.keys(t).join(", ") + "}"
        ), e = e(), t.current = e, function() {
          t.current = null;
        };
    }
    function wr(e, t, a) {
      typeof t != "function" && console.error(
        "Expected useImperativeHandle() second argument to be a function that creates a handle. Instead received: %s.",
        t !== null ? typeof t : "null"
      ), a = a != null ? a.concat([e]) : null;
      var i = 4194308;
      (qe.mode & Ku) !== Bt && (i |= 134217728), Ka(
        i,
        la,
        $p.bind(null, t, e),
        a
      );
    }
    function Gn(e, t, a) {
      typeof t != "function" && console.error(
        "Expected useImperativeHandle() second argument to be a function that creates a handle. Instead received: %s.",
        t !== null ? typeof t : "null"
      ), a = a != null ? a.concat([e]) : null, sl(
        4,
        la,
        $p.bind(null, t, e),
        a
      );
    }
    function Jf(e, t) {
      return Qt().memoizedState = [
        e,
        t === void 0 ? null : t
      ], e;
    }
    function dc(e, t) {
      var a = ot();
      t = t === void 0 ? null : t;
      var i = a.memoizedState;
      return t !== null && hi(t, i[1]) ? i[0] : (a.memoizedState = [e, t], e);
    }
    function qr(e, t) {
      var a = Qt();
      t = t === void 0 ? null : t;
      var i = e();
      if (Js) {
        oe(!0);
        try {
          e();
        } finally {
          oe(!1);
        }
      }
      return a.memoizedState = [i, t], i;
    }
    function vi(e, t) {
      var a = ot();
      t = t === void 0 ? null : t;
      var i = a.memoizedState;
      if (t !== null && hi(t, i[1]))
        return i[0];
      if (i = e(), Js) {
        oe(!0);
        try {
          e();
        } finally {
          oe(!1);
        }
      }
      return a.memoizedState = [i, t], i;
    }
    function jr(e, t) {
      var a = Qt();
      return Yr(a, e, t);
    }
    function kf(e, t) {
      var a = ot();
      return $f(
        a,
        Ut.memoizedState,
        e,
        t
      );
    }
    function Br(e, t) {
      var a = ot();
      return Ut === null ? Yr(a, e, t) : $f(
        a,
        Ut.memoizedState,
        e,
        t
      );
    }
    function Yr(e, t, a) {
      return a === void 0 || (cf & 1073741824) !== 0 ? e.memoizedState = t : (e.memoizedState = a, e = uv(), qe.lanes |= e, sf |= e, a);
    }
    function $f(e, t, a, i) {
      return Ha(a, t) ? a : dh.current !== null ? (e = Yr(e, a, i), Ha(e, t) || (Kl = !0), e) : (cf & 42) === 0 ? (Kl = !0, e.memoizedState = a) : (e = uv(), qe.lanes |= e, sf |= e, t);
    }
    function yy(e, t, a, i, o) {
      var f = Ce.p;
      Ce.p = f !== 0 && f < En ? f : En;
      var d = L.T, h = {};
      L.T = h, Vr(e, !1, t, a), h._updatedFibers = /* @__PURE__ */ new Set();
      try {
        var v = o(), b = L.S;
        if (b !== null && b(h, v), v !== null && typeof v == "object" && typeof v.then == "function") {
          var B = Zp(
            v,
            i
          );
          Uu(
            e,
            t,
            B,
            ha(e)
          );
        } else
          Uu(
            e,
            t,
            i,
            ha(e)
          );
      } catch (X) {
        Uu(
          e,
          t,
          { then: function() {
          }, status: "rejected", reason: X },
          ha(e)
        );
      } finally {
        Ce.p = f, L.T = d, d === null && h._updatedFibers && (e = h._updatedFibers.size, h._updatedFibers.clear(), 10 < e && console.warn(
          "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
        ));
      }
    }
    function hc(e, t, a, i) {
      if (e.tag !== 5)
        throw Error(
          "Expected the form instance to be a HostComponent. This is a bug in React."
        );
      var o = my(e).queue;
      yy(
        e,
        o,
        t,
        nr,
        a === null ? te : function() {
          return py(e), a(i);
        }
      );
    }
    function my(e) {
      var t = e.memoizedState;
      if (t !== null) return t;
      t = {
        memoizedState: nr,
        baseState: nr,
        baseQueue: null,
        queue: {
          pending: null,
          lanes: 0,
          dispatch: null,
          lastRenderedReducer: dt,
          lastRenderedState: nr
        },
        next: null
      };
      var a = {};
      return t.next = {
        memoizedState: a,
        baseState: a,
        baseQueue: null,
        queue: {
          pending: null,
          lanes: 0,
          dispatch: null,
          lastRenderedReducer: dt,
          lastRenderedState: a
        },
        next: null
      }, e.memoizedState = t, e = e.alternate, e !== null && (e.memoizedState = t), t;
    }
    function py(e) {
      L.T === null && console.error(
        "requestFormReset was called outside a transition or action. To fix, move to an action, or wrap with startTransition."
      );
      var t = my(e).next.queue;
      Uu(
        e,
        t,
        {},
        ha(e)
      );
    }
    function Ln() {
      var e = Xf(!1);
      return e = yy.bind(
        null,
        qe,
        e.queue,
        !0,
        !1
      ), Qt().memoizedState = e, [!1, e];
    }
    function Gr() {
      var e = Qa(dt)[0], t = ot().memoizedState;
      return [
        typeof e == "boolean" ? e : sc(e),
        t
      ];
    }
    function Lr() {
      var e = rc(dt)[0], t = ot().memoizedState;
      return [
        typeof e == "boolean" ? e : sc(e),
        t
      ];
    }
    function sa() {
      return Ct(gp);
    }
    function Vn() {
      var e = Qt(), t = xt.identifierPrefix;
      if (mt) {
        var a = Lc, i = Gc;
        a = (i & ~(1 << 32 - Ql(i) - 1)).toString(32) + a, t = "«" + t + "R" + a, a = Kv++, 0 < a && (t += "H" + a.toString(32)), t += "»";
      } else
        a = YS++, t = "«" + t + "r" + a.toString(32) + "»";
      return e.memoizedState = t;
    }
    function yc() {
      return Qt().memoizedState = vy.bind(
        null,
        qe
      );
    }
    function vy(e, t) {
      for (var a = e.return; a !== null; ) {
        switch (a.tag) {
          case 24:
          case 3:
            var i = ha(a);
            e = wn(i);
            var o = hn(a, e, i);
            o !== null && (Kt(o, a, i), di(o, a, i)), a = Bf(), t != null && o !== null && console.error(
              "The seed argument is not enabled outside experimental channels."
            ), e.payload = { cache: a };
            return;
        }
        a = a.return;
      }
    }
    function gy(e, t, a) {
      var i = arguments;
      typeof i[3] == "function" && console.error(
        "State updates from the useState() and useReducer() Hooks don't support the second callback argument. To execute a side effect after rendering, declare it in the component body with useEffect()."
      ), i = ha(e);
      var o = {
        lane: i,
        revertLane: 0,
        action: a,
        hasEagerState: !1,
        eagerState: null,
        next: null
      };
      Wf(e) ? mc(t, o) : (o = Kh(e, t, o, i), o !== null && (Kt(o, e, i), Ff(o, t, i))), zn(e, i);
    }
    function Ro(e, t, a) {
      var i = arguments;
      typeof i[3] == "function" && console.error(
        "State updates from the useState() and useReducer() Hooks don't support the second callback argument. To execute a side effect after rendering, declare it in the component body with useEffect()."
      ), i = ha(e), Uu(e, t, a, i), zn(e, i);
    }
    function Uu(e, t, a, i) {
      var o = {
        lane: i,
        revertLane: 0,
        action: a,
        hasEagerState: !1,
        eagerState: null,
        next: null
      };
      if (Wf(e)) mc(t, o);
      else {
        var f = e.alternate;
        if (e.lanes === 0 && (f === null || f.lanes === 0) && (f = t.lastRenderedReducer, f !== null)) {
          var d = L.H;
          L.H = ku;
          try {
            var h = t.lastRenderedState, v = f(h, a);
            if (o.hasEagerState = !0, o.eagerState = v, Ha(v, h))
              return Er(e, t, o, 0), xt === null && Hf(), !1;
          } catch {
          } finally {
            L.H = d;
          }
        }
        if (a = Kh(e, t, o, i), a !== null)
          return Kt(a, e, i), Ff(a, t, i), !0;
      }
      return !1;
    }
    function Vr(e, t, a, i) {
      if (L.T === null && Qs === 0 && console.error(
        "An optimistic state update occurred outside a transition or action. To fix, move the update to an action, or wrap with startTransition."
      ), i = {
        lane: 2,
        revertLane: $y(),
        action: i,
        hasEagerState: !1,
        eagerState: null,
        next: null
      }, Wf(e)) {
        if (t)
          throw Error("Cannot update optimistic state while rendering.");
        console.error("Cannot call startTransition while rendering.");
      } else
        t = Kh(
          e,
          a,
          i,
          2
        ), t !== null && Kt(t, e, 2);
      zn(e, 2);
    }
    function Wf(e) {
      var t = e.alternate;
      return e === qe || t !== null && t === qe;
    }
    function mc(e, t) {
      yh = Zv = !0;
      var a = e.pending;
      a === null ? t.next = t : (t.next = a.next, a.next = t), e.pending = t;
    }
    function Ff(e, t, a) {
      if ((a & 4194048) !== 0) {
        var i = t.lanes;
        i &= e.pendingLanes, a |= i, t.lanes = a, Pu(e, a);
      }
    }
    function bl(e) {
      var t = Fe;
      return e != null && (Fe = t === null ? e : t.concat(e)), t;
    }
    function Oo(e, t, a) {
      for (var i = Object.keys(e.props), o = 0; o < i.length; o++) {
        var f = i[o];
        if (f !== "children" && f !== "key") {
          t === null && (t = qf(e, a.mode, 0), t._debugInfo = Fe, t.return = a), ye(
            t,
            function(d) {
              console.error(
                "Invalid prop `%s` supplied to `React.Fragment`. React.Fragment can only have `key` and `children` props.",
                d
              );
            },
            f
          );
          break;
        }
      }
    }
    function Do(e) {
      var t = lp;
      return lp += 1, mh === null && (mh = cy()), La(mh, e, t);
    }
    function Ja(e, t) {
      t = t.props.ref, e.ref = t !== void 0 ? t : null;
    }
    function Ge(e, t) {
      throw t.$$typeof === zs ? Error(
        `A React Element from an older version of React was rendered. This is not supported. It can happen if:
- Multiple copies of the "react" package is used.
- A library pre-bundled an old copy of "react" or "react/jsx-runtime".
- A compiler tries to "inline" JSX instead of using the runtime.`
      ) : (e = Object.prototype.toString.call(t), Error(
        "Objects are not valid as a React child (found: " + (e === "[object Object]" ? "object with keys {" + Object.keys(t).join(", ") + "}" : e) + "). If you meant to render a collection of children, use an array instead."
      ));
    }
    function vt(e, t) {
      var a = re(e) || "Component";
      q1[a] || (q1[a] = !0, t = t.displayName || t.name || "Component", e.tag === 3 ? console.error(
        `Functions are not valid as a React child. This may happen if you return %s instead of <%s /> from render. Or maybe you meant to call this function rather than return it.
  root.render(%s)`,
        t,
        t,
        t
      ) : console.error(
        `Functions are not valid as a React child. This may happen if you return %s instead of <%s /> from render. Or maybe you meant to call this function rather than return it.
  <%s>{%s}</%s>`,
        t,
        t,
        a,
        t,
        a
      ));
    }
    function Vt(e, t) {
      var a = re(e) || "Component";
      j1[a] || (j1[a] = !0, t = String(t), e.tag === 3 ? console.error(
        `Symbols are not valid as a React child.
  root.render(%s)`,
        t
      ) : console.error(
        `Symbols are not valid as a React child.
  <%s>%s</%s>`,
        a,
        t,
        a
      ));
    }
    function If(e) {
      function t(T, E) {
        if (e) {
          var A = T.deletions;
          A === null ? (T.deletions = [E], T.flags |= 16) : A.push(E);
        }
      }
      function a(T, E) {
        if (!e) return null;
        for (; E !== null; )
          t(T, E), E = E.sibling;
        return null;
      }
      function i(T) {
        for (var E = /* @__PURE__ */ new Map(); T !== null; )
          T.key !== null ? E.set(T.key, T) : E.set(T.index, T), T = T.sibling;
        return E;
      }
      function o(T, E) {
        return T = Cn(T, E), T.index = 0, T.sibling = null, T;
      }
      function f(T, E, A) {
        return T.index = A, e ? (A = T.alternate, A !== null ? (A = A.index, A < E ? (T.flags |= 67108866, E) : A) : (T.flags |= 67108866, E)) : (T.flags |= 1048576, E);
      }
      function d(T) {
        return e && T.alternate === null && (T.flags |= 67108866), T;
      }
      function h(T, E, A, Z) {
        return E === null || E.tag !== 6 ? (E = ii(
          A,
          T.mode,
          Z
        ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Fe, E) : (E = o(E, A), E.return = T, E._debugInfo = Fe, E);
      }
      function v(T, E, A, Z) {
        var ne = A.type;
        return ne === Le ? (E = B(
          T,
          E,
          A.props.children,
          Z,
          A.key
        ), Oo(A, E, T), E) : E !== null && (E.elementType === ne || Lp(E, A) || typeof ne == "object" && ne !== null && ne.$$typeof === Ca && of(ne) === E.type) ? (E = o(E, A.props), Ja(E, A), E.return = T, E._debugOwner = A._owner, E._debugInfo = Fe, E) : (E = qf(A, T.mode, Z), Ja(E, A), E.return = T, E._debugInfo = Fe, E);
      }
      function b(T, E, A, Z) {
        return E === null || E.tag !== 4 || E.stateNode.containerInfo !== A.containerInfo || E.stateNode.implementation !== A.implementation ? (E = Wh(A, T.mode, Z), E.return = T, E._debugInfo = Fe, E) : (E = o(E, A.children || []), E.return = T, E._debugInfo = Fe, E);
      }
      function B(T, E, A, Z, ne) {
        return E === null || E.tag !== 7 ? (E = ui(
          A,
          T.mode,
          Z,
          ne
        ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Fe, E) : (E = o(E, A), E.return = T, E._debugInfo = Fe, E);
      }
      function X(T, E, A) {
        if (typeof E == "string" && E !== "" || typeof E == "number" || typeof E == "bigint")
          return E = ii(
            "" + E,
            T.mode,
            A
          ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Fe, E;
        if (typeof E == "object" && E !== null) {
          switch (E.$$typeof) {
            case Ui:
              return A = qf(
                E,
                T.mode,
                A
              ), Ja(A, E), A.return = T, T = bl(E._debugInfo), A._debugInfo = Fe, Fe = T, A;
            case Nc:
              return E = Wh(
                E,
                T.mode,
                A
              ), E.return = T, E._debugInfo = Fe, E;
            case Ca:
              var Z = bl(E._debugInfo);
              return E = of(E), T = X(T, E, A), Fe = Z, T;
          }
          if (we(E) || Ue(E))
            return A = ui(
              E,
              T.mode,
              A,
              null
            ), A.return = T, A._debugOwner = T, A._debugTask = T._debugTask, T = bl(E._debugInfo), A._debugInfo = Fe, Fe = T, A;
          if (typeof E.then == "function")
            return Z = bl(E._debugInfo), T = X(
              T,
              Do(E),
              A
            ), Fe = Z, T;
          if (E.$$typeof === Ia)
            return X(
              T,
              jf(T, E),
              A
            );
          Ge(T, E);
        }
        return typeof E == "function" && vt(T, E), typeof E == "symbol" && Vt(T, E), null;
      }
      function N(T, E, A, Z) {
        var ne = E !== null ? E.key : null;
        if (typeof A == "string" && A !== "" || typeof A == "number" || typeof A == "bigint")
          return ne !== null ? null : h(T, E, "" + A, Z);
        if (typeof A == "object" && A !== null) {
          switch (A.$$typeof) {
            case Ui:
              return A.key === ne ? (ne = bl(A._debugInfo), T = v(
                T,
                E,
                A,
                Z
              ), Fe = ne, T) : null;
            case Nc:
              return A.key === ne ? b(T, E, A, Z) : null;
            case Ca:
              return ne = bl(A._debugInfo), A = of(A), T = N(
                T,
                E,
                A,
                Z
              ), Fe = ne, T;
          }
          if (we(A) || Ue(A))
            return ne !== null ? null : (ne = bl(A._debugInfo), T = B(
              T,
              E,
              A,
              Z,
              null
            ), Fe = ne, T);
          if (typeof A.then == "function")
            return ne = bl(A._debugInfo), T = N(
              T,
              E,
              Do(A),
              Z
            ), Fe = ne, T;
          if (A.$$typeof === Ia)
            return N(
              T,
              E,
              jf(T, A),
              Z
            );
          Ge(T, A);
        }
        return typeof A == "function" && vt(T, A), typeof A == "symbol" && Vt(T, A), null;
      }
      function Q(T, E, A, Z, ne) {
        if (typeof Z == "string" && Z !== "" || typeof Z == "number" || typeof Z == "bigint")
          return T = T.get(A) || null, h(E, T, "" + Z, ne);
        if (typeof Z == "object" && Z !== null) {
          switch (Z.$$typeof) {
            case Ui:
              return A = T.get(
                Z.key === null ? A : Z.key
              ) || null, T = bl(Z._debugInfo), E = v(
                E,
                A,
                Z,
                ne
              ), Fe = T, E;
            case Nc:
              return T = T.get(
                Z.key === null ? A : Z.key
              ) || null, b(E, T, Z, ne);
            case Ca:
              var Ve = bl(Z._debugInfo);
              return Z = of(Z), E = Q(
                T,
                E,
                A,
                Z,
                ne
              ), Fe = Ve, E;
          }
          if (we(Z) || Ue(Z))
            return A = T.get(A) || null, T = bl(Z._debugInfo), E = B(
              E,
              A,
              Z,
              ne,
              null
            ), Fe = T, E;
          if (typeof Z.then == "function")
            return Ve = bl(Z._debugInfo), E = Q(
              T,
              E,
              A,
              Do(Z),
              ne
            ), Fe = Ve, E;
          if (Z.$$typeof === Ia)
            return Q(
              T,
              E,
              A,
              jf(E, Z),
              ne
            );
          Ge(E, Z);
        }
        return typeof Z == "function" && vt(E, Z), typeof Z == "symbol" && Vt(E, Z), null;
      }
      function me(T, E, A, Z) {
        if (typeof A != "object" || A === null) return Z;
        switch (A.$$typeof) {
          case Ui:
          case Nc:
            et(T, E, A);
            var ne = A.key;
            if (typeof ne != "string") break;
            if (Z === null) {
              Z = /* @__PURE__ */ new Set(), Z.add(ne);
              break;
            }
            if (!Z.has(ne)) {
              Z.add(ne);
              break;
            }
            ye(E, function() {
              console.error(
                "Encountered two children with the same key, `%s`. Keys should be unique so that components maintain their identity across updates. Non-unique keys may cause children to be duplicated and/or omitted — the behavior is unsupported and could change in a future version.",
                ne
              );
            });
            break;
          case Ca:
            A = of(A), me(T, E, A, Z);
        }
        return Z;
      }
      function xe(T, E, A, Z) {
        for (var ne = null, Ve = null, pe = null, Xe = E, Ze = E = 0, Yt = null; Xe !== null && Ze < A.length; Ze++) {
          Xe.index > Ze ? (Yt = Xe, Xe = null) : Yt = Xe.sibling;
          var yl = N(
            T,
            Xe,
            A[Ze],
            Z
          );
          if (yl === null) {
            Xe === null && (Xe = Yt);
            break;
          }
          ne = me(
            T,
            yl,
            A[Ze],
            ne
          ), e && Xe && yl.alternate === null && t(T, Xe), E = f(yl, E, Ze), pe === null ? Ve = yl : pe.sibling = yl, pe = yl, Xe = Yt;
        }
        if (Ze === A.length)
          return a(T, Xe), mt && nc(T, Ze), Ve;
        if (Xe === null) {
          for (; Ze < A.length; Ze++)
            Xe = X(T, A[Ze], Z), Xe !== null && (ne = me(
              T,
              Xe,
              A[Ze],
              ne
            ), E = f(
              Xe,
              E,
              Ze
            ), pe === null ? Ve = Xe : pe.sibling = Xe, pe = Xe);
          return mt && nc(T, Ze), Ve;
        }
        for (Xe = i(Xe); Ze < A.length; Ze++)
          Yt = Q(
            Xe,
            T,
            Ze,
            A[Ze],
            Z
          ), Yt !== null && (ne = me(
            T,
            Yt,
            A[Ze],
            ne
          ), e && Yt.alternate !== null && Xe.delete(
            Yt.key === null ? Ze : Yt.key
          ), E = f(
            Yt,
            E,
            Ze
          ), pe === null ? Ve = Yt : pe.sibling = Yt, pe = Yt);
        return e && Xe.forEach(function(Wc) {
          return t(T, Wc);
        }), mt && nc(T, Ze), Ve;
      }
      function Ht(T, E, A, Z) {
        if (A == null)
          throw Error("An iterable object provided no iterator.");
        for (var ne = null, Ve = null, pe = E, Xe = E = 0, Ze = null, Yt = null, yl = A.next(); pe !== null && !yl.done; Xe++, yl = A.next()) {
          pe.index > Xe ? (Ze = pe, pe = null) : Ze = pe.sibling;
          var Wc = N(T, pe, yl.value, Z);
          if (Wc === null) {
            pe === null && (pe = Ze);
            break;
          }
          Yt = me(
            T,
            Wc,
            yl.value,
            Yt
          ), e && pe && Wc.alternate === null && t(T, pe), E = f(Wc, E, Xe), Ve === null ? ne = Wc : Ve.sibling = Wc, Ve = Wc, pe = Ze;
        }
        if (yl.done)
          return a(T, pe), mt && nc(T, Xe), ne;
        if (pe === null) {
          for (; !yl.done; Xe++, yl = A.next())
            pe = X(T, yl.value, Z), pe !== null && (Yt = me(
              T,
              pe,
              yl.value,
              Yt
            ), E = f(
              pe,
              E,
              Xe
            ), Ve === null ? ne = pe : Ve.sibling = pe, Ve = pe);
          return mt && nc(T, Xe), ne;
        }
        for (pe = i(pe); !yl.done; Xe++, yl = A.next())
          Ze = Q(
            pe,
            T,
            Xe,
            yl.value,
            Z
          ), Ze !== null && (Yt = me(
            T,
            Ze,
            yl.value,
            Yt
          ), e && Ze.alternate !== null && pe.delete(
            Ze.key === null ? Xe : Ze.key
          ), E = f(
            Ze,
            E,
            Xe
          ), Ve === null ? ne = Ze : Ve.sibling = Ze, Ve = Ze);
        return e && pe.forEach(function(dT) {
          return t(T, dT);
        }), mt && nc(T, Xe), ne;
      }
      function ft(T, E, A, Z) {
        if (typeof A == "object" && A !== null && A.type === Le && A.key === null && (Oo(A, null, T), A = A.props.children), typeof A == "object" && A !== null) {
          switch (A.$$typeof) {
            case Ui:
              var ne = bl(A._debugInfo);
              e: {
                for (var Ve = A.key; E !== null; ) {
                  if (E.key === Ve) {
                    if (Ve = A.type, Ve === Le) {
                      if (E.tag === 7) {
                        a(
                          T,
                          E.sibling
                        ), Z = o(
                          E,
                          A.props.children
                        ), Z.return = T, Z._debugOwner = A._owner, Z._debugInfo = Fe, Oo(A, Z, T), T = Z;
                        break e;
                      }
                    } else if (E.elementType === Ve || Lp(
                      E,
                      A
                    ) || typeof Ve == "object" && Ve !== null && Ve.$$typeof === Ca && of(Ve) === E.type) {
                      a(
                        T,
                        E.sibling
                      ), Z = o(E, A.props), Ja(Z, A), Z.return = T, Z._debugOwner = A._owner, Z._debugInfo = Fe, T = Z;
                      break e;
                    }
                    a(T, E);
                    break;
                  } else t(T, E);
                  E = E.sibling;
                }
                A.type === Le ? (Z = ui(
                  A.props.children,
                  T.mode,
                  Z,
                  A.key
                ), Z.return = T, Z._debugOwner = T, Z._debugTask = T._debugTask, Z._debugInfo = Fe, Oo(A, Z, T), T = Z) : (Z = qf(
                  A,
                  T.mode,
                  Z
                ), Ja(Z, A), Z.return = T, Z._debugInfo = Fe, T = Z);
              }
              return T = d(T), Fe = ne, T;
            case Nc:
              e: {
                for (ne = A, A = ne.key; E !== null; ) {
                  if (E.key === A)
                    if (E.tag === 4 && E.stateNode.containerInfo === ne.containerInfo && E.stateNode.implementation === ne.implementation) {
                      a(
                        T,
                        E.sibling
                      ), Z = o(
                        E,
                        ne.children || []
                      ), Z.return = T, T = Z;
                      break e;
                    } else {
                      a(T, E);
                      break;
                    }
                  else t(T, E);
                  E = E.sibling;
                }
                Z = Wh(
                  ne,
                  T.mode,
                  Z
                ), Z.return = T, T = Z;
              }
              return d(T);
            case Ca:
              return ne = bl(A._debugInfo), A = of(A), T = ft(
                T,
                E,
                A,
                Z
              ), Fe = ne, T;
          }
          if (we(A))
            return ne = bl(A._debugInfo), T = xe(
              T,
              E,
              A,
              Z
            ), Fe = ne, T;
          if (Ue(A)) {
            if (ne = bl(A._debugInfo), Ve = Ue(A), typeof Ve != "function")
              throw Error(
                "An object is not an iterable. This error is likely caused by a bug in React. Please file an issue."
              );
            var pe = Ve.call(A);
            return pe === A ? (T.tag !== 0 || Object.prototype.toString.call(T.type) !== "[object GeneratorFunction]" || Object.prototype.toString.call(pe) !== "[object Generator]") && (N1 || console.error(
              "Using Iterators as children is unsupported and will likely yield unexpected results because enumerating a generator mutates it. You may convert it to an array with `Array.from()` or the `[...spread]` operator before rendering. You can also use an Iterable that can iterate multiple times over the same items."
            ), N1 = !0) : A.entries !== Ve || o0 || (console.error(
              "Using Maps as children is not supported. Use an array of keyed ReactElements instead."
            ), o0 = !0), T = Ht(
              T,
              E,
              pe,
              Z
            ), Fe = ne, T;
          }
          if (typeof A.then == "function")
            return ne = bl(A._debugInfo), T = ft(
              T,
              E,
              Do(A),
              Z
            ), Fe = ne, T;
          if (A.$$typeof === Ia)
            return ft(
              T,
              E,
              jf(T, A),
              Z
            );
          Ge(T, A);
        }
        return typeof A == "string" && A !== "" || typeof A == "number" || typeof A == "bigint" ? (ne = "" + A, E !== null && E.tag === 6 ? (a(
          T,
          E.sibling
        ), Z = o(E, ne), Z.return = T, T = Z) : (a(T, E), Z = ii(
          ne,
          T.mode,
          Z
        ), Z.return = T, Z._debugOwner = T, Z._debugTask = T._debugTask, Z._debugInfo = Fe, T = Z), d(T)) : (typeof A == "function" && vt(T, A), typeof A == "symbol" && Vt(T, A), a(T, E));
      }
      return function(T, E, A, Z) {
        var ne = Fe;
        Fe = null;
        try {
          lp = 0;
          var Ve = ft(
            T,
            E,
            A,
            Z
          );
          return mh = null, Ve;
        } catch (Yt) {
          if (Yt === Im || Yt === Vv) throw Yt;
          var pe = z(29, Yt, null, T.mode);
          pe.lanes = Z, pe.return = T;
          var Xe = pe._debugInfo = Fe;
          if (pe._debugOwner = T._debugOwner, pe._debugTask = T._debugTask, Xe != null) {
            for (var Ze = Xe.length - 1; 0 <= Ze; Ze--)
              if (typeof Xe[Ze].stack == "string") {
                pe._debugOwner = Xe[Ze], pe._debugTask = Xe[Ze].debugTask;
                break;
              }
          }
          return pe;
        } finally {
          Fe = ne;
        }
      };
    }
    function Da(e) {
      var t = e.alternate;
      ze(
        Yl,
        Yl.current & vh,
        e
      ), ze(mu, e, e), Li === null && (t === null || dh.current !== null || t.memoizedState !== null) && (Li = e);
    }
    function gi(e) {
      if (e.tag === 22) {
        if (ze(Yl, Yl.current, e), ze(mu, e, e), Li === null) {
          var t = e.alternate;
          t !== null && t.memoizedState !== null && (Li = e);
        }
      } else vn(e);
    }
    function vn(e) {
      ze(Yl, Yl.current, e), ze(
        mu,
        mu.current,
        e
      );
    }
    function za(e) {
      Se(mu, e), Li === e && (Li = null), Se(Yl, e);
    }
    function Cu(e) {
      for (var t = e; t !== null; ) {
        if (t.tag === 13) {
          var a = t.memoizedState;
          if (a !== null && (a = a.dehydrated, a === null || a.data === Jc || tu(a)))
            return t;
        } else if (t.tag === 19 && t.memoizedProps.revealOrder !== void 0) {
          if ((t.flags & 128) !== 0) return t;
        } else if (t.child !== null) {
          t.child.return = t, t = t.child;
          continue;
        }
        if (t === e) break;
        for (; t.sibling === null; ) {
          if (t.return === null || t.return === e) return null;
          t = t.return;
        }
        t.sibling.return = t.return, t = t.sibling;
      }
      return null;
    }
    function by(e) {
      if (e !== null && typeof e != "function") {
        var t = String(e);
        $1.has(t) || ($1.add(t), console.error(
          "Expected the last optional `callback` argument to be a function. Instead received: %s.",
          e
        ));
      }
    }
    function Xt(e, t, a, i) {
      var o = e.memoizedState, f = a(i, o);
      if (e.mode & Sa) {
        oe(!0);
        try {
          f = a(i, o);
        } finally {
          oe(!1);
        }
      }
      f === void 0 && (t = De(t) || "Component", Z1.has(t) || (Z1.add(t), console.error(
        "%s.getDerivedStateFromProps(): A valid state object (or null) must be returned. You have returned undefined.",
        t
      ))), o = f == null ? o : Je({}, o, f), e.memoizedState = o, e.lanes === 0 && (e.updateQueue.baseState = o);
    }
    function Xr(e, t, a, i, o, f, d) {
      var h = e.stateNode;
      if (typeof h.shouldComponentUpdate == "function") {
        if (a = h.shouldComponentUpdate(
          i,
          f,
          d
        ), e.mode & Sa) {
          oe(!0);
          try {
            a = h.shouldComponentUpdate(
              i,
              f,
              d
            );
          } finally {
            oe(!1);
          }
        }
        return a === void 0 && console.error(
          "%s.shouldComponentUpdate(): Returned undefined instead of a boolean value. Make sure to return true or false.",
          De(t) || "Component"
        ), a;
      }
      return t.prototype && t.prototype.isPureReactComponent ? !xf(a, i) || !xf(o, f) : !0;
    }
    function Qr(e, t, a, i) {
      var o = t.state;
      typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps(a, i), typeof t.UNSAFE_componentWillReceiveProps == "function" && t.UNSAFE_componentWillReceiveProps(a, i), t.state !== o && (e = re(e) || "Component", G1.has(e) || (G1.add(e), console.error(
        "%s.componentWillReceiveProps(): Assigning directly to this.state is deprecated (except inside a component's constructor). Use setState instead.",
        e
      )), f0.enqueueReplaceState(
        t,
        t.state,
        null
      ));
    }
    function bi(e, t) {
      var a = t;
      if ("ref" in t) {
        a = {};
        for (var i in t)
          i !== "ref" && (a[i] = t[i]);
      }
      if (e = e.defaultProps) {
        a === t && (a = Je({}, a));
        for (var o in e)
          a[o] === void 0 && (a[o] = e[o]);
      }
      return a;
    }
    function Sy(e) {
      s0(e), console.warn(
        `%s

%s
`,
        gh ? "An error occurred in the <" + gh + "> component." : "An error occurred in one of your React components.",
        `Consider adding an error boundary to your tree to customize error handling behavior.
Visit https://react.dev/link/error-boundaries to learn more about error boundaries.`
      );
    }
    function Wp(e) {
      var t = gh ? "The above error occurred in the <" + gh + "> component." : "The above error occurred in one of your React components.", a = "React will try to recreate this component tree from scratch using the error boundary you provided, " + ((r0 || "Anonymous") + ".");
      if (typeof e == "object" && e !== null && typeof e.environmentName == "string") {
        var i = e.environmentName;
        e = [
          `%o

%s

%s
`,
          e,
          t,
          a
        ].slice(0), typeof e[0] == "string" ? e.splice(
          0,
          1,
          zb + e[0],
          Mb,
          dg + i + dg,
          _b
        ) : e.splice(
          0,
          0,
          zb,
          Mb,
          dg + i + dg,
          _b
        ), e.unshift(console), i = sT.apply(console.error, e), i();
      } else
        console.error(
          `%o

%s

%s
`,
          e,
          t,
          a
        );
    }
    function Zr(e) {
      s0(e);
    }
    function zo(e, t) {
      try {
        gh = t.source ? re(t.source) : null, r0 = null;
        var a = t.value;
        if (L.actQueue !== null)
          L.thrownErrors.push(a);
        else {
          var i = e.onUncaughtError;
          i(a, { componentStack: t.stack });
        }
      } catch (o) {
        setTimeout(function() {
          throw o;
        });
      }
    }
    function Kr(e, t, a) {
      try {
        gh = a.source ? re(a.source) : null, r0 = re(t);
        var i = e.onCaughtError;
        i(a.value, {
          componentStack: a.stack,
          errorBoundary: t.tag === 1 ? t.stateNode : null
        });
      } catch (o) {
        setTimeout(function() {
          throw o;
        });
      }
    }
    function Ll(e, t, a) {
      return a = wn(a), a.tag = e0, a.payload = { element: null }, a.callback = function() {
        ye(t.source, zo, e, t);
      }, a;
    }
    function Zt(e) {
      return e = wn(e), e.tag = e0, e;
    }
    function Pf(e, t, a, i) {
      var o = a.type.getDerivedStateFromError;
      if (typeof o == "function") {
        var f = i.value;
        e.payload = function() {
          return o(f);
        }, e.callback = function() {
          Vp(a), ye(
            i.source,
            Kr,
            t,
            a,
            i
          );
        };
      }
      var d = a.stateNode;
      d !== null && typeof d.componentDidCatch == "function" && (e.callback = function() {
        Vp(a), ye(
          i.source,
          Kr,
          t,
          a,
          i
        ), typeof o != "function" && (df === null ? df = /* @__PURE__ */ new Set([this]) : df.add(this)), LS(this, i), typeof o == "function" || (a.lanes & 2) === 0 && console.error(
          "%s: Error boundaries should implement getDerivedStateFromError(). In that method, return a state update to display an error message or fallback UI.",
          re(a) || "Unknown"
        );
      });
    }
    function es(e, t, a, i, o) {
      if (a.flags |= 32768, Ft && wo(e, o), i !== null && typeof i == "object" && typeof i.then == "function") {
        if (t = a.alternate, t !== null && Ul(
          t,
          a,
          o,
          !0
        ), mt && (Vc = !0), a = mu.current, a !== null) {
          switch (a.tag) {
            case 13:
              return Li === null ? dd() : a.alternate === null && ul === Kc && (ul = m0), a.flags &= -257, a.flags |= 65536, a.lanes = o, i === Pg ? a.flags |= 16384 : (t = a.updateQueue, t === null ? a.updateQueue = /* @__PURE__ */ new Set([i]) : t.add(i), Zy(e, i, o)), !1;
            case 22:
              return a.flags |= 65536, i === Pg ? a.flags |= 16384 : (t = a.updateQueue, t === null ? (t = {
                transitions: null,
                markerInstances: null,
                retryQueue: /* @__PURE__ */ new Set([i])
              }, a.updateQueue = t) : (a = t.retryQueue, a === null ? t.retryQueue = /* @__PURE__ */ new Set([i]) : a.add(i)), Zy(e, i, o)), !1;
          }
          throw Error(
            "Unexpected Suspense handler tag (" + a.tag + "). This is a bug in React."
          );
        }
        return Zy(e, i, o), dd(), !1;
      }
      if (mt)
        return Vc = !0, t = mu.current, t !== null ? ((t.flags & 65536) === 0 && (t.flags |= 256), t.flags |= 65536, t.lanes = o, i !== $g && so(
          Ra(
            Error(
              "There was an error while hydrating but React was able to recover by instead client rendering from the nearest Suspense boundary.",
              { cause: i }
            ),
            a
          )
        )) : (i !== $g && so(
          Ra(
            Error(
              "There was an error while hydrating but React was able to recover by instead client rendering the entire root.",
              { cause: i }
            ),
            a
          )
        ), e = e.current.alternate, e.flags |= 65536, o &= -o, e.lanes |= o, i = Ra(i, a), o = Ll(
          e.stateNode,
          i,
          o
        ), ho(e, o), ul !== ks && (ul = Eh)), !1;
      var f = Ra(
        Error(
          "There was an error during concurrent rendering but React was able to recover by instead synchronously rendering the entire root.",
          { cause: i }
        ),
        a
      );
      if (sp === null ? sp = [f] : sp.push(f), ul !== ks && (ul = Eh), t === null) return !0;
      i = Ra(i, a), a = t;
      do {
        switch (a.tag) {
          case 3:
            return a.flags |= 65536, e = o & -o, a.lanes |= e, e = Ll(
              a.stateNode,
              i,
              e
            ), ho(a, e), !1;
          case 1:
            if (t = a.type, f = a.stateNode, (a.flags & 128) === 0 && (typeof t.getDerivedStateFromError == "function" || f !== null && typeof f.componentDidCatch == "function" && (df === null || !df.has(f))))
              return a.flags |= 65536, o &= -o, a.lanes |= o, o = Zt(o), Pf(
                o,
                e,
                a,
                i
              ), ho(a, o), !1;
        }
        a = a.return;
      } while (a !== null);
      return !1;
    }
    function al(e, t, a, i) {
      t.child = e === null ? B1(t, null, a, i) : ph(
        t,
        e.child,
        a,
        i
      );
    }
    function Jr(e, t, a, i, o) {
      a = a.render;
      var f = t.ref;
      if ("ref" in i) {
        var d = {};
        for (var h in i)
          h !== "ref" && (d[h] = i[h]);
      } else d = i;
      return fi(t), wt(t), i = yi(
        e,
        t,
        a,
        d,
        f,
        o
      ), h = fa(), na(), e !== null && !Kl ? (Du(e, t, o), Qn(e, t, o)) : (mt && h && Or(t), t.flags |= 1, al(e, t, i, o), t.child);
    }
    function Xn(e, t, a, i, o) {
      if (e === null) {
        var f = a.type;
        return typeof f == "function" && !kh(f) && f.defaultProps === void 0 && a.compare === null ? (a = ac(f), t.tag = 15, t.type = a, Fr(t, f), ts(
          e,
          t,
          a,
          i,
          o
        )) : (e = Rr(
          a.type,
          null,
          i,
          t,
          t.mode,
          o
        ), e.ref = t.ref, e.return = t, t.child = e);
      }
      if (f = e.child, !ad(e, o)) {
        var d = f.memoizedProps;
        if (a = a.compare, a = a !== null ? a : xf, a(d, i) && e.ref === t.ref)
          return Qn(
            e,
            t,
            o
          );
      }
      return t.flags |= 1, e = Cn(f, i), e.ref = t.ref, e.return = t, t.child = e;
    }
    function ts(e, t, a, i, o) {
      if (e !== null) {
        var f = e.memoizedProps;
        if (xf(f, i) && e.ref === t.ref && t.type === e.type)
          if (Kl = !1, t.pendingProps = i = f, ad(e, o))
            (e.flags & 131072) !== 0 && (Kl = !0);
          else
            return t.lanes = e.lanes, Qn(e, t, o);
      }
      return Wr(
        e,
        t,
        a,
        i,
        o
      );
    }
    function kr(e, t, a) {
      var i = t.pendingProps, o = i.children, f = e !== null ? e.memoizedState : null;
      if (i.mode === "hidden") {
        if ((t.flags & 128) !== 0) {
          if (i = f !== null ? f.baseLanes | a : a, e !== null) {
            for (o = t.child = e.child, f = 0; o !== null; )
              f = f | o.lanes | o.childLanes, o = o.sibling;
            t.childLanes = f & ~i;
          } else t.childLanes = 0, t.child = null;
          return $r(
            e,
            t,
            i,
            a
          );
        }
        if ((a & 536870912) !== 0)
          t.memoizedState = { baseLanes: 0, cachePool: null }, e !== null && _r(
            t,
            f !== null ? f.cachePool : null
          ), f !== null ? oa(t, f) : Gf(t), gi(t);
        else
          return t.lanes = t.childLanes = 536870912, $r(
            e,
            t,
            f !== null ? f.baseLanes | a : a,
            a
          );
      } else
        f !== null ? (_r(t, f.cachePool), oa(t, f), vn(t), t.memoizedState = null) : (e !== null && _r(t, null), Gf(t), vn(t));
      return al(e, t, o, a), t.child;
    }
    function $r(e, t, a, i) {
      var o = iy();
      return o = o === null ? null : {
        parent: jl._currentValue,
        pool: o
      }, t.memoizedState = {
        baseLanes: a,
        cachePool: o
      }, e !== null && _r(t, null), Gf(t), gi(t), e !== null && Ul(e, t, i, !0), null;
    }
    function ls(e, t) {
      var a = t.ref;
      if (a === null)
        e !== null && e.ref !== null && (t.flags |= 4194816);
      else {
        if (typeof a != "function" && typeof a != "object")
          throw Error(
            "Expected ref to be a function, an object returned by React.createRef(), or undefined/null."
          );
        (e === null || e.ref !== a) && (t.flags |= 4194816);
      }
    }
    function Wr(e, t, a, i, o) {
      if (a.prototype && typeof a.prototype.render == "function") {
        var f = De(a) || "Unknown";
        F1[f] || (console.error(
          "The <%s /> component appears to have a render method, but doesn't extend React.Component. This is likely to cause errors. Change %s to extend React.Component instead.",
          f,
          f
        ), F1[f] = !0);
      }
      return t.mode & Sa && Ju.recordLegacyContextWarning(
        t,
        null
      ), e === null && (Fr(t, t.type), a.contextTypes && (f = De(a) || "Unknown", P1[f] || (P1[f] = !0, console.error(
        "%s uses the legacy contextTypes API which was removed in React 19. Use React.createContext() with React.useContext() instead. (https://react.dev/link/legacy-context)",
        f
      )))), fi(t), wt(t), a = yi(
        e,
        t,
        a,
        i,
        void 0,
        o
      ), i = fa(), na(), e !== null && !Kl ? (Du(e, t, o), Qn(e, t, o)) : (mt && i && Or(t), t.flags |= 1, al(e, t, a, o), t.child);
    }
    function Ty(e, t, a, i, o, f) {
      return fi(t), wt(t), Qc = -1, tp = e !== null && e.type !== t.type, t.updateQueue = null, a = vo(
        t,
        i,
        a,
        o
      ), Lf(e, t), i = fa(), na(), e !== null && !Kl ? (Du(e, t, f), Qn(e, t, f)) : (mt && i && Or(t), t.flags |= 1, al(e, t, a, f), t.child);
    }
    function Ey(e, t, a, i, o) {
      switch (Qe(t)) {
        case !1:
          var f = t.stateNode, d = new t.type(
            t.memoizedProps,
            f.context
          ).state;
          f.updater.enqueueSetState(f, d, null);
          break;
        case !0:
          t.flags |= 128, t.flags |= 65536, f = Error("Simulated error coming from DevTools");
          var h = o & -o;
          if (t.lanes |= h, d = xt, d === null)
            throw Error(
              "Expected a work-in-progress root. This is a bug in React. Please file an issue."
            );
          h = Zt(h), Pf(
            h,
            d,
            t,
            Ra(f, t)
          ), ho(t, h);
      }
      if (fi(t), t.stateNode === null) {
        if (d = nf, f = a.contextType, "contextType" in a && f !== null && (f === void 0 || f.$$typeof !== Ia) && !k1.has(a) && (k1.add(a), h = f === void 0 ? " However, it is set to undefined. This can be caused by a typo or by mixing up named and default imports. This can also happen due to a circular dependency, so try moving the createContext() call to a separate file." : typeof f != "object" ? " However, it is set to a " + typeof f + "." : f.$$typeof === Yd ? " Did you accidentally pass the Context.Consumer instead?" : " However, it is set to an object with keys {" + Object.keys(f).join(", ") + "}.", console.error(
          "%s defines an invalid contextType. contextType should point to the Context object returned by React.createContext().%s",
          De(a) || "Component",
          h
        )), typeof f == "object" && f !== null && (d = Ct(f)), f = new a(i, d), t.mode & Sa) {
          oe(!0);
          try {
            f = new a(i, d);
          } finally {
            oe(!1);
          }
        }
        if (d = t.memoizedState = f.state !== null && f.state !== void 0 ? f.state : null, f.updater = f0, t.stateNode = f, f._reactInternals = t, f._reactInternalInstance = Y1, typeof a.getDerivedStateFromProps == "function" && d === null && (d = De(a) || "Component", L1.has(d) || (L1.add(d), console.error(
          "`%s` uses `getDerivedStateFromProps` but its initial state is %s. This is not recommended. Instead, define the initial state by assigning an object to `this.state` in the constructor of `%s`. This ensures that `getDerivedStateFromProps` arguments have a consistent shape.",
          d,
          f.state === null ? "null" : "undefined",
          d
        ))), typeof a.getDerivedStateFromProps == "function" || typeof f.getSnapshotBeforeUpdate == "function") {
          var v = h = d = null;
          if (typeof f.componentWillMount == "function" && f.componentWillMount.__suppressDeprecationWarning !== !0 ? d = "componentWillMount" : typeof f.UNSAFE_componentWillMount == "function" && (d = "UNSAFE_componentWillMount"), typeof f.componentWillReceiveProps == "function" && f.componentWillReceiveProps.__suppressDeprecationWarning !== !0 ? h = "componentWillReceiveProps" : typeof f.UNSAFE_componentWillReceiveProps == "function" && (h = "UNSAFE_componentWillReceiveProps"), typeof f.componentWillUpdate == "function" && f.componentWillUpdate.__suppressDeprecationWarning !== !0 ? v = "componentWillUpdate" : typeof f.UNSAFE_componentWillUpdate == "function" && (v = "UNSAFE_componentWillUpdate"), d !== null || h !== null || v !== null) {
            f = De(a) || "Component";
            var b = typeof a.getDerivedStateFromProps == "function" ? "getDerivedStateFromProps()" : "getSnapshotBeforeUpdate()";
            X1.has(f) || (X1.add(f), console.error(
              `Unsafe legacy lifecycles will not be called for components using new component APIs.

%s uses %s but also contains the following legacy lifecycles:%s%s%s

The above lifecycles should be removed. Learn more about this warning here:
https://react.dev/link/unsafe-component-lifecycles`,
              f,
              b,
              d !== null ? `
  ` + d : "",
              h !== null ? `
  ` + h : "",
              v !== null ? `
  ` + v : ""
            ));
          }
        }
        f = t.stateNode, d = De(a) || "Component", f.render || (a.prototype && typeof a.prototype.render == "function" ? console.error(
          "No `render` method found on the %s instance: did you accidentally return an object from the constructor?",
          d
        ) : console.error(
          "No `render` method found on the %s instance: you may have forgotten to define `render`.",
          d
        )), !f.getInitialState || f.getInitialState.isReactClassApproved || f.state || console.error(
          "getInitialState was defined on %s, a plain JavaScript class. This is only supported for classes created using React.createClass. Did you mean to define a state property instead?",
          d
        ), f.getDefaultProps && !f.getDefaultProps.isReactClassApproved && console.error(
          "getDefaultProps was defined on %s, a plain JavaScript class. This is only supported for classes created using React.createClass. Use a static property to define defaultProps instead.",
          d
        ), f.contextType && console.error(
          "contextType was defined as an instance property on %s. Use a static property to define contextType instead.",
          d
        ), a.childContextTypes && !J1.has(a) && (J1.add(a), console.error(
          "%s uses the legacy childContextTypes API which was removed in React 19. Use React.createContext() instead. (https://react.dev/link/legacy-context)",
          d
        )), a.contextTypes && !K1.has(a) && (K1.add(a), console.error(
          "%s uses the legacy contextTypes API which was removed in React 19. Use React.createContext() with static contextType instead. (https://react.dev/link/legacy-context)",
          d
        )), typeof f.componentShouldUpdate == "function" && console.error(
          "%s has a method called componentShouldUpdate(). Did you mean shouldComponentUpdate()? The name is phrased as a question because the function is expected to return a value.",
          d
        ), a.prototype && a.prototype.isPureReactComponent && typeof f.shouldComponentUpdate < "u" && console.error(
          "%s has a method called shouldComponentUpdate(). shouldComponentUpdate should not be used when extending React.PureComponent. Please extend React.Component if shouldComponentUpdate is used.",
          De(a) || "A pure component"
        ), typeof f.componentDidUnmount == "function" && console.error(
          "%s has a method called componentDidUnmount(). But there is no such lifecycle method. Did you mean componentWillUnmount()?",
          d
        ), typeof f.componentDidReceiveProps == "function" && console.error(
          "%s has a method called componentDidReceiveProps(). But there is no such lifecycle method. If you meant to update the state in response to changing props, use componentWillReceiveProps(). If you meant to fetch data or run side-effects or mutations after React has updated the UI, use componentDidUpdate().",
          d
        ), typeof f.componentWillRecieveProps == "function" && console.error(
          "%s has a method called componentWillRecieveProps(). Did you mean componentWillReceiveProps()?",
          d
        ), typeof f.UNSAFE_componentWillRecieveProps == "function" && console.error(
          "%s has a method called UNSAFE_componentWillRecieveProps(). Did you mean UNSAFE_componentWillReceiveProps()?",
          d
        ), h = f.props !== i, f.props !== void 0 && h && console.error(
          "When calling super() in `%s`, make sure to pass up the same props that your component's constructor was passed.",
          d
        ), f.defaultProps && console.error(
          "Setting defaultProps as an instance property on %s is not supported and will be ignored. Instead, define defaultProps as a static property on %s.",
          d,
          d
        ), typeof f.getSnapshotBeforeUpdate != "function" || typeof f.componentDidUpdate == "function" || V1.has(a) || (V1.add(a), console.error(
          "%s: getSnapshotBeforeUpdate() should be used with componentDidUpdate(). This component defines getSnapshotBeforeUpdate() only.",
          De(a)
        )), typeof f.getDerivedStateFromProps == "function" && console.error(
          "%s: getDerivedStateFromProps() is defined as an instance method and will be ignored. Instead, declare it as a static method.",
          d
        ), typeof f.getDerivedStateFromError == "function" && console.error(
          "%s: getDerivedStateFromError() is defined as an instance method and will be ignored. Instead, declare it as a static method.",
          d
        ), typeof a.getSnapshotBeforeUpdate == "function" && console.error(
          "%s: getSnapshotBeforeUpdate() is defined as a static method and will be ignored. Instead, declare it as an instance method.",
          d
        ), (h = f.state) && (typeof h != "object" || we(h)) && console.error("%s.state: must be set to an object or null", d), typeof f.getChildContext == "function" && typeof a.childContextTypes != "object" && console.error(
          "%s.getChildContext(): childContextTypes must be defined in order to use getChildContext().",
          d
        ), f = t.stateNode, f.props = i, f.state = t.memoizedState, f.refs = {}, ca(t), d = a.contextType, f.context = typeof d == "object" && d !== null ? Ct(d) : nf, f.state === i && (d = De(a) || "Component", Q1.has(d) || (Q1.add(d), console.error(
          "%s: It is not recommended to assign props directly to state because updates to props won't be reflected in state. In most cases, it is better to use props directly.",
          d
        ))), t.mode & Sa && Ju.recordLegacyContextWarning(
          t,
          f
        ), Ju.recordUnsafeLifecycleWarnings(
          t,
          f
        ), f.state = t.memoizedState, d = a.getDerivedStateFromProps, typeof d == "function" && (Xt(
          t,
          a,
          d,
          i
        ), f.state = t.memoizedState), typeof a.getDerivedStateFromProps == "function" || typeof f.getSnapshotBeforeUpdate == "function" || typeof f.UNSAFE_componentWillMount != "function" && typeof f.componentWillMount != "function" || (d = f.state, typeof f.componentWillMount == "function" && f.componentWillMount(), typeof f.UNSAFE_componentWillMount == "function" && f.UNSAFE_componentWillMount(), d !== f.state && (console.error(
          "%s.componentWillMount(): Assigning directly to this.state is deprecated (except inside a component's constructor). Use setState instead.",
          re(t) || "Component"
        ), f0.enqueueReplaceState(
          f,
          f.state,
          null
        )), yo(t, i, f, o), qn(), f.state = t.memoizedState), typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ku) !== Bt && (t.flags |= 134217728), f = !0;
      } else if (e === null) {
        f = t.stateNode;
        var B = t.memoizedProps;
        h = bi(a, B), f.props = h;
        var X = f.context;
        v = a.contextType, d = nf, typeof v == "object" && v !== null && (d = Ct(v)), b = a.getDerivedStateFromProps, v = typeof b == "function" || typeof f.getSnapshotBeforeUpdate == "function", B = t.pendingProps !== B, v || typeof f.UNSAFE_componentWillReceiveProps != "function" && typeof f.componentWillReceiveProps != "function" || (B || X !== d) && Qr(
          t,
          f,
          i,
          d
        ), uf = !1;
        var N = t.memoizedState;
        f.state = N, yo(t, i, f, o), qn(), X = t.memoizedState, B || N !== X || uf ? (typeof b == "function" && (Xt(
          t,
          a,
          b,
          i
        ), X = t.memoizedState), (h = uf || Xr(
          t,
          a,
          h,
          i,
          N,
          X,
          d
        )) ? (v || typeof f.UNSAFE_componentWillMount != "function" && typeof f.componentWillMount != "function" || (typeof f.componentWillMount == "function" && f.componentWillMount(), typeof f.UNSAFE_componentWillMount == "function" && f.UNSAFE_componentWillMount()), typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ku) !== Bt && (t.flags |= 134217728)) : (typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ku) !== Bt && (t.flags |= 134217728), t.memoizedProps = i, t.memoizedState = X), f.props = i, f.state = X, f.context = d, f = h) : (typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ku) !== Bt && (t.flags |= 134217728), f = !1);
      } else {
        f = t.stateNode, ri(e, t), d = t.memoizedProps, v = bi(a, d), f.props = v, b = t.pendingProps, N = f.context, X = a.contextType, h = nf, typeof X == "object" && X !== null && (h = Ct(X)), B = a.getDerivedStateFromProps, (X = typeof B == "function" || typeof f.getSnapshotBeforeUpdate == "function") || typeof f.UNSAFE_componentWillReceiveProps != "function" && typeof f.componentWillReceiveProps != "function" || (d !== b || N !== h) && Qr(
          t,
          f,
          i,
          h
        ), uf = !1, N = t.memoizedState, f.state = N, yo(t, i, f, o), qn();
        var Q = t.memoizedState;
        d !== b || N !== Q || uf || e !== null && e.dependencies !== null && oi(e.dependencies) ? (typeof B == "function" && (Xt(
          t,
          a,
          B,
          i
        ), Q = t.memoizedState), (v = uf || Xr(
          t,
          a,
          v,
          i,
          N,
          Q,
          h
        ) || e !== null && e.dependencies !== null && oi(e.dependencies)) ? (X || typeof f.UNSAFE_componentWillUpdate != "function" && typeof f.componentWillUpdate != "function" || (typeof f.componentWillUpdate == "function" && f.componentWillUpdate(i, Q, h), typeof f.UNSAFE_componentWillUpdate == "function" && f.UNSAFE_componentWillUpdate(
          i,
          Q,
          h
        )), typeof f.componentDidUpdate == "function" && (t.flags |= 4), typeof f.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof f.componentDidUpdate != "function" || d === e.memoizedProps && N === e.memoizedState || (t.flags |= 4), typeof f.getSnapshotBeforeUpdate != "function" || d === e.memoizedProps && N === e.memoizedState || (t.flags |= 1024), t.memoizedProps = i, t.memoizedState = Q), f.props = i, f.state = Q, f.context = h, f = v) : (typeof f.componentDidUpdate != "function" || d === e.memoizedProps && N === e.memoizedState || (t.flags |= 4), typeof f.getSnapshotBeforeUpdate != "function" || d === e.memoizedProps && N === e.memoizedState || (t.flags |= 1024), f = !1);
      }
      if (h = f, ls(e, t), d = (t.flags & 128) !== 0, h || d) {
        if (h = t.stateNode, Ef(t), d && typeof a.getDerivedStateFromError != "function")
          a = null, tn = -1;
        else {
          if (wt(t), a = R1(h), t.mode & Sa) {
            oe(!0);
            try {
              R1(h);
            } finally {
              oe(!1);
            }
          }
          na();
        }
        t.flags |= 1, e !== null && d ? (t.child = ph(
          t,
          e.child,
          null,
          o
        ), t.child = ph(
          t,
          null,
          a,
          o
        )) : al(e, t, a, o), t.memoizedState = h.state, e = t.child;
      } else
        e = Qn(
          e,
          t,
          o
        );
      return o = t.stateNode, f && o.props !== i && (bh || console.error(
        "It looks like %s is reassigning its own `this.props` while rendering. This is not supported and can lead to confusing bugs.",
        re(t) || "a component"
      ), bh = !0), e;
    }
    function Ay(e, t, a, i) {
      return ic(), t.flags |= 256, al(e, t, a, i), t.child;
    }
    function Fr(e, t) {
      t && t.childContextTypes && console.error(
        `childContextTypes cannot be defined on a function component.
  %s.childContextTypes = ...`,
        t.displayName || t.name || "Component"
      ), typeof t.getDerivedStateFromProps == "function" && (e = De(t) || "Unknown", eb[e] || (console.error(
        "%s: Function components do not support getDerivedStateFromProps.",
        e
      ), eb[e] = !0)), typeof t.contextType == "object" && t.contextType !== null && (t = De(t) || "Unknown", I1[t] || (console.error(
        "%s: Function components do not support contextType.",
        t
      ), I1[t] = !0));
    }
    function as(e) {
      return { baseLanes: e, cachePool: Kp() };
    }
    function Ir(e, t, a) {
      return e = e !== null ? e.childLanes & ~a : 0, t && (e |= On), e;
    }
    function Fp(e, t, a) {
      var i, o = t.pendingProps;
      Ee(t) && (t.flags |= 128);
      var f = !1, d = (t.flags & 128) !== 0;
      if ((i = d) || (i = e !== null && e.memoizedState === null ? !1 : (Yl.current & ap) !== 0), i && (f = !0, t.flags &= -129), i = (t.flags & 32) !== 0, t.flags &= -33, e === null) {
        if (mt) {
          if (f ? Da(t) : vn(t), mt) {
            var h = nl, v;
            if (!(v = !h)) {
              e: {
                var b = h;
                for (v = Yi; b.nodeType !== 8; ) {
                  if (!v) {
                    v = null;
                    break e;
                  }
                  if (b = Nl(b.nextSibling), b === null) {
                    v = null;
                    break e;
                  }
                }
                v = b;
              }
              v !== null ? (fn(), t.memoizedState = {
                dehydrated: v,
                treeContext: Ls !== null ? { id: Gc, overflow: Lc } : null,
                retryLane: 536870912,
                hydrationErrors: null
              }, b = z(18, null, null, Bt), b.stateNode = v, b.return = t, t.child = b, Na = t, nl = null, v = !0) : v = !1, v = !v;
            }
            v && (Fh(
              t,
              h
            ), xn(t));
          }
          if (h = t.memoizedState, h !== null && (h = h.dehydrated, h !== null))
            return tu(h) ? t.lanes = 32 : t.lanes = 536870912, null;
          za(t);
        }
        return h = o.children, o = o.fallback, f ? (vn(t), f = t.mode, h = ns(
          {
            mode: "hidden",
            children: h
          },
          f
        ), o = ui(
          o,
          f,
          a,
          null
        ), h.return = t, o.return = t, h.sibling = o, t.child = h, f = t.child, f.memoizedState = as(a), f.childLanes = Ir(
          e,
          i,
          a
        ), t.memoizedState = h0, o) : (Da(t), Pr(
          t,
          h
        ));
      }
      var B = e.memoizedState;
      if (B !== null && (h = B.dehydrated, h !== null)) {
        if (d)
          t.flags & 256 ? (Da(t), t.flags &= -257, t = ed(
            e,
            t,
            a
          )) : t.memoizedState !== null ? (vn(t), t.child = e.child, t.flags |= 128, t = null) : (vn(t), f = o.fallback, h = t.mode, o = ns(
            {
              mode: "visible",
              children: o.children
            },
            h
          ), f = ui(
            f,
            h,
            a,
            null
          ), f.flags |= 2, o.return = t, f.return = t, o.sibling = f, t.child = o, ph(
            t,
            e.child,
            null,
            a
          ), o = t.child, o.memoizedState = as(a), o.childLanes = Ir(
            e,
            i,
            a
          ), t.memoizedState = h0, t = f);
        else if (Da(t), mt && console.error(
          "We should not be hydrating here. This is a bug in React. Please file a bug."
        ), tu(h)) {
          if (i = h.nextSibling && h.nextSibling.dataset, i) {
            v = i.dgst;
            var X = i.msg;
            b = i.stck;
            var N = i.cstck;
          }
          h = X, i = v, o = b, v = f = N, f = Error(h || "The server could not finish this Suspense boundary, likely due to an error during server rendering. Switched to client rendering."), f.stack = o || "", f.digest = i, i = v === void 0 ? null : v, o = {
            value: f,
            source: null,
            stack: i
          }, typeof i == "string" && Jg.set(
            f,
            o
          ), so(o), t = ed(
            e,
            t,
            a
          );
        } else if (Kl || Ul(
          e,
          t,
          a,
          !1
        ), i = (a & e.childLanes) !== 0, Kl || i) {
          if (i = xt, i !== null && (o = a & -a, o = (o & 42) !== 0 ? 1 : Ol(
            o
          ), o = (o & (i.suspendedLanes | a)) !== 0 ? 0 : o, o !== 0 && o !== B.retryLane))
            throw B.retryLane = o, ia(
              e,
              o
            ), Kt(
              i,
              e,
              o
            ), W1;
          h.data === Jc || dd(), t = ed(
            e,
            t,
            a
          );
        } else
          h.data === Jc ? (t.flags |= 192, t.child = e.child, t = null) : (e = B.treeContext, nl = Nl(
            h.nextSibling
          ), Na = t, mt = !0, Vs = null, Vc = !1, ru = null, Yi = !1, e !== null && (fn(), fu[su++] = Gc, fu[su++] = Lc, fu[su++] = Ls, Gc = e.id, Lc = e.overflow, Ls = t), t = Pr(
            t,
            o.children
          ), t.flags |= 4096);
        return t;
      }
      return f ? (vn(t), f = o.fallback, h = t.mode, v = e.child, b = v.sibling, o = Cn(
        v,
        {
          mode: "hidden",
          children: o.children
        }
      ), o.subtreeFlags = v.subtreeFlags & 65011712, b !== null ? f = Cn(
        b,
        f
      ) : (f = ui(
        f,
        h,
        a,
        null
      ), f.flags |= 2), f.return = t, o.return = t, o.sibling = f, t.child = o, o = f, f = t.child, h = e.child.memoizedState, h === null ? h = as(a) : (v = h.cachePool, v !== null ? (b = jl._currentValue, v = v.parent !== b ? { parent: b, pool: b } : v) : v = Kp(), h = {
        baseLanes: h.baseLanes | a,
        cachePool: v
      }), f.memoizedState = h, f.childLanes = Ir(
        e,
        i,
        a
      ), t.memoizedState = h0, o) : (Da(t), a = e.child, e = a.sibling, a = Cn(a, {
        mode: "visible",
        children: o.children
      }), a.return = t, a.sibling = null, e !== null && (i = t.deletions, i === null ? (t.deletions = [e], t.flags |= 16) : i.push(e)), t.child = a, t.memoizedState = null, a);
    }
    function Pr(e, t) {
      return t = ns(
        { mode: "visible", children: t },
        e.mode
      ), t.return = e, e.child = t;
    }
    function ns(e, t) {
      return e = z(22, e, null, t), e.lanes = 0, e.stateNode = {
        _visibility: Nv,
        _pendingMarkers: null,
        _retryCache: null,
        _transitions: null
      }, e;
    }
    function ed(e, t, a) {
      return ph(t, e.child, null, a), e = Pr(
        t,
        t.pendingProps.children
      ), e.flags |= 2, t.memoizedState = null, e;
    }
    function td(e, t, a) {
      e.lanes |= t;
      var i = e.alternate;
      i !== null && (i.lanes |= t), ly(
        e.return,
        t,
        a
      );
    }
    function Ry(e, t) {
      var a = we(e);
      return e = !a && typeof Ue(e) == "function", a || e ? (a = a ? "array" : "iterable", console.error(
        "A nested %s was passed to row #%s in <SuspenseList />. Wrap it in an additional SuspenseList to configure its revealOrder: <SuspenseList revealOrder=...> ... <SuspenseList revealOrder=...>{%s}</SuspenseList> ... </SuspenseList>",
        a,
        t,
        a
      ), !1) : !0;
    }
    function ld(e, t, a, i, o) {
      var f = e.memoizedState;
      f === null ? e.memoizedState = {
        isBackwards: t,
        rendering: null,
        renderingStartTime: 0,
        last: i,
        tail: a,
        tailMode: o
      } : (f.isBackwards = t, f.rendering = null, f.renderingStartTime = 0, f.last = i, f.tail = a, f.tailMode = o);
    }
    function Oy(e, t, a) {
      var i = t.pendingProps, o = i.revealOrder, f = i.tail;
      if (i = i.children, o !== void 0 && o !== "forwards" && o !== "backwards" && o !== "together" && !tb[o])
        if (tb[o] = !0, typeof o == "string")
          switch (o.toLowerCase()) {
            case "together":
            case "forwards":
            case "backwards":
              console.error(
                '"%s" is not a valid value for revealOrder on <SuspenseList />. Use lowercase "%s" instead.',
                o,
                o.toLowerCase()
              );
              break;
            case "forward":
            case "backward":
              console.error(
                '"%s" is not a valid value for revealOrder on <SuspenseList />. React uses the -s suffix in the spelling. Use "%ss" instead.',
                o,
                o.toLowerCase()
              );
              break;
            default:
              console.error(
                '"%s" is not a supported revealOrder on <SuspenseList />. Did you mean "together", "forwards" or "backwards"?',
                o
              );
          }
        else
          console.error(
            '%s is not a supported value for revealOrder on <SuspenseList />. Did you mean "together", "forwards" or "backwards"?',
            o
          );
      f === void 0 || d0[f] || (f !== "collapsed" && f !== "hidden" ? (d0[f] = !0, console.error(
        '"%s" is not a supported value for tail on <SuspenseList />. Did you mean "collapsed" or "hidden"?',
        f
      )) : o !== "forwards" && o !== "backwards" && (d0[f] = !0, console.error(
        '<SuspenseList tail="%s" /> is only valid if revealOrder is "forwards" or "backwards". Did you mean to specify revealOrder="forwards"?',
        f
      )));
      e: if ((o === "forwards" || o === "backwards") && i !== void 0 && i !== null && i !== !1)
        if (we(i)) {
          for (var d = 0; d < i.length; d++)
            if (!Ry(i[d], d)) break e;
        } else if (d = Ue(i), typeof d == "function") {
          if (d = d.call(i))
            for (var h = d.next(), v = 0; !h.done; h = d.next()) {
              if (!Ry(h.value, v)) break e;
              v++;
            }
        } else
          console.error(
            'A single row was passed to a <SuspenseList revealOrder="%s" />. This is not useful since it needs multiple rows. Did you mean to pass multiple children or an array?',
            o
          );
      if (al(e, t, i, a), i = Yl.current, (i & ap) !== 0)
        i = i & vh | ap, t.flags |= 128;
      else {
        if (e !== null && (e.flags & 128) !== 0)
          e: for (e = t.child; e !== null; ) {
            if (e.tag === 13)
              e.memoizedState !== null && td(
                e,
                a,
                t
              );
            else if (e.tag === 19)
              td(e, a, t);
            else if (e.child !== null) {
              e.child.return = e, e = e.child;
              continue;
            }
            if (e === t) break e;
            for (; e.sibling === null; ) {
              if (e.return === null || e.return === t)
                break e;
              e = e.return;
            }
            e.sibling.return = e.return, e = e.sibling;
          }
        i &= vh;
      }
      switch (ze(Yl, i, t), o) {
        case "forwards":
          for (a = t.child, o = null; a !== null; )
            e = a.alternate, e !== null && Cu(e) === null && (o = a), a = a.sibling;
          a = o, a === null ? (o = t.child, t.child = null) : (o = a.sibling, a.sibling = null), ld(
            t,
            !1,
            o,
            a,
            f
          );
          break;
        case "backwards":
          for (a = null, o = t.child, t.child = null; o !== null; ) {
            if (e = o.alternate, e !== null && Cu(e) === null) {
              t.child = o;
              break;
            }
            e = o.sibling, o.sibling = a, a = o, o = e;
          }
          ld(
            t,
            !0,
            a,
            null,
            f
          );
          break;
        case "together":
          ld(t, !1, null, null, void 0);
          break;
        default:
          t.memoizedState = null;
      }
      return t.child;
    }
    function Qn(e, t, a) {
      if (e !== null && (t.dependencies = e.dependencies), tn = -1, sf |= t.lanes, (a & t.childLanes) === 0)
        if (e !== null) {
          if (Ul(
            e,
            t,
            a,
            !1
          ), (a & t.childLanes) === 0)
            return null;
        } else return null;
      if (e !== null && t.child !== e.child)
        throw Error("Resuming work not yet implemented.");
      if (t.child !== null) {
        for (e = t.child, a = Cn(e, e.pendingProps), t.child = a, a.return = t; e.sibling !== null; )
          e = e.sibling, a = a.sibling = Cn(e, e.pendingProps), a.return = t;
        a.sibling = null;
      }
      return t.child;
    }
    function ad(e, t) {
      return (e.lanes & t) !== 0 ? !0 : (e = e.dependencies, !!(e !== null && oi(e)));
    }
    function Mg(e, t, a) {
      switch (t.tag) {
        case 3:
          Gt(
            t,
            t.stateNode.containerInfo
          ), ci(
            t,
            jl,
            e.memoizedState.cache
          ), ic();
          break;
        case 27:
        case 5:
          F(t);
          break;
        case 4:
          Gt(
            t,
            t.stateNode.containerInfo
          );
          break;
        case 10:
          ci(
            t,
            t.type,
            t.memoizedProps.value
          );
          break;
        case 12:
          (a & t.childLanes) !== 0 && (t.flags |= 4), t.flags |= 2048;
          var i = t.stateNode;
          i.effectDuration = -0, i.passiveEffectDuration = -0;
          break;
        case 13:
          if (i = t.memoizedState, i !== null)
            return i.dehydrated !== null ? (Da(t), t.flags |= 128, null) : (a & t.child.childLanes) !== 0 ? Fp(
              e,
              t,
              a
            ) : (Da(t), e = Qn(
              e,
              t,
              a
            ), e !== null ? e.sibling : null);
          Da(t);
          break;
        case 19:
          var o = (e.flags & 128) !== 0;
          if (i = (a & t.childLanes) !== 0, i || (Ul(
            e,
            t,
            a,
            !1
          ), i = (a & t.childLanes) !== 0), o) {
            if (i)
              return Oy(
                e,
                t,
                a
              );
            t.flags |= 128;
          }
          if (o = t.memoizedState, o !== null && (o.rendering = null, o.tail = null, o.lastEffect = null), ze(
            Yl,
            Yl.current,
            t
          ), i) break;
          return null;
        case 22:
        case 23:
          return t.lanes = 0, kr(e, t, a);
        case 24:
          ci(
            t,
            jl,
            e.memoizedState.cache
          );
      }
      return Qn(e, t, a);
    }
    function nd(e, t, a) {
      if (t._debugNeedsRemount && e !== null) {
        a = Rr(
          t.type,
          t.key,
          t.pendingProps,
          t._debugOwner || null,
          t.mode,
          t.lanes
        ), a._debugStack = t._debugStack, a._debugTask = t._debugTask;
        var i = t.return;
        if (i === null) throw Error("Cannot swap the root fiber.");
        if (e.alternate = null, t.alternate = null, a.index = t.index, a.sibling = t.sibling, a.return = t.return, a.ref = t.ref, a._debugInfo = t._debugInfo, t === i.child)
          i.child = a;
        else {
          var o = i.child;
          if (o === null)
            throw Error("Expected parent to have a child.");
          for (; o.sibling !== t; )
            if (o = o.sibling, o === null)
              throw Error("Expected to find the previous sibling.");
          o.sibling = a;
        }
        return t = i.deletions, t === null ? (i.deletions = [e], i.flags |= 16) : t.push(e), a.flags |= 2, a;
      }
      if (e !== null)
        if (e.memoizedProps !== t.pendingProps || t.type !== e.type)
          Kl = !0;
        else {
          if (!ad(e, a) && (t.flags & 128) === 0)
            return Kl = !1, Mg(
              e,
              t,
              a
            );
          Kl = (e.flags & 131072) !== 0;
        }
      else
        Kl = !1, (i = mt) && (fn(), i = (t.flags & 1048576) !== 0), i && (i = t.index, fn(), Xp(t, qv, i));
      switch (t.lanes = 0, t.tag) {
        case 16:
          e: if (i = t.pendingProps, e = of(t.elementType), t.type = e, typeof e == "function")
            kh(e) ? (i = bi(
              e,
              i
            ), t.tag = 1, t.type = e = ac(e), t = Ey(
              null,
              t,
              e,
              i,
              a
            )) : (t.tag = 0, Fr(t, e), t.type = e = ac(e), t = Wr(
              null,
              t,
              e,
              i,
              a
            ));
          else {
            if (e != null) {
              if (o = e.$$typeof, o === Yu) {
                t.tag = 11, t.type = e = Jh(e), t = Jr(
                  null,
                  t,
                  e,
                  i,
                  a
                );
                break e;
              } else if (o === Ms) {
                t.tag = 14, t = Xn(
                  null,
                  t,
                  e,
                  i,
                  a
                );
                break e;
              }
            }
            throw t = "", e !== null && typeof e == "object" && e.$$typeof === Ca && (t = " Did you wrap a component in React.lazy() more than once?"), e = De(e) || e, Error(
              "Element type is invalid. Received a promise that resolves to: " + e + ". Lazy element type must resolve to a class or function." + t
            );
          }
          return t;
        case 0:
          return Wr(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 1:
          return i = t.type, o = bi(
            i,
            t.pendingProps
          ), Ey(
            e,
            t,
            i,
            o,
            a
          );
        case 3:
          e: {
            if (Gt(
              t,
              t.stateNode.containerInfo
            ), e === null)
              throw Error(
                "Should have a current fiber. This is a bug in React."
              );
            i = t.pendingProps;
            var f = t.memoizedState;
            o = f.element, ri(e, t), yo(t, i, null, a);
            var d = t.memoizedState;
            if (i = d.cache, ci(t, jl, i), i !== f.cache && ay(
              t,
              [jl],
              a,
              !0
            ), qn(), i = d.element, f.isDehydrated)
              if (f = {
                element: i,
                isDehydrated: !1,
                cache: d.cache
              }, t.updateQueue.baseState = f, t.memoizedState = f, t.flags & 256) {
                t = Ay(
                  e,
                  t,
                  i,
                  a
                );
                break e;
              } else if (i !== o) {
                o = Ra(
                  Error(
                    "This root received an early update, before anything was able hydrate. Switched the entire root to client rendering."
                  ),
                  t
                ), so(o), t = Ay(
                  e,
                  t,
                  i,
                  a
                );
                break e;
              } else {
                switch (e = t.stateNode.containerInfo, e.nodeType) {
                  case 9:
                    e = e.body;
                    break;
                  default:
                    e = e.nodeName === "HTML" ? e.ownerDocument.body : e;
                }
                for (nl = Nl(e.firstChild), Na = t, mt = !0, Vs = null, Vc = !1, ru = null, Yi = !0, e = B1(
                  t,
                  null,
                  i,
                  a
                ), t.child = e; e; )
                  e.flags = e.flags & -3 | 4096, e = e.sibling;
              }
            else {
              if (ic(), i === o) {
                t = Qn(
                  e,
                  t,
                  a
                );
                break e;
              }
              al(
                e,
                t,
                i,
                a
              );
            }
            t = t.child;
          }
          return t;
        case 26:
          return ls(e, t), e === null ? (e = Bu(
            t.type,
            null,
            t.pendingProps,
            null
          )) ? t.memoizedState = e : mt || (e = t.type, a = t.pendingProps, i = Ot(
            au.current
          ), i = lt(
            i
          ).createElement(e), i[Zl] = t, i[ga] = a, kt(i, e, a), D(i), t.stateNode = i) : t.memoizedState = Bu(
            t.type,
            e.memoizedProps,
            t.pendingProps,
            e.memoizedState
          ), null;
        case 27:
          return F(t), e === null && mt && (i = Ot(au.current), o = O(), i = t.stateNode = sm(
            t.type,
            t.pendingProps,
            i,
            o,
            !1
          ), Vc || (o = _t(
            i,
            t.type,
            t.pendingProps,
            o
          ), o !== null && (sn(t, 0).serverProps = o)), Na = t, Yi = !0, o = nl, eu(t.type) ? (q0 = o, nl = Nl(
            i.firstChild
          )) : nl = o), al(
            e,
            t,
            t.pendingProps.children,
            a
          ), ls(e, t), e === null && (t.flags |= 4194304), t.child;
        case 5:
          return e === null && mt && (f = O(), i = mr(
            t.type,
            f.ancestorInfo
          ), o = nl, (d = !o) || (d = Di(
            o,
            t.type,
            t.pendingProps,
            Yi
          ), d !== null ? (t.stateNode = d, Vc || (f = _t(
            d,
            t.type,
            t.pendingProps,
            f
          ), f !== null && (sn(t, 0).serverProps = f)), Na = t, nl = Nl(
            d.firstChild
          ), Yi = !1, f = !0) : f = !1, d = !f), d && (i && Fh(t, o), xn(t))), F(t), o = t.type, f = t.pendingProps, d = e !== null ? e.memoizedProps : null, i = f.children, Pn(o, f) ? i = null : d !== null && Pn(o, d) && (t.flags |= 32), t.memoizedState !== null && (o = yi(
            e,
            t,
            Xa,
            null,
            null,
            a
          ), gp._currentValue = o), ls(e, t), al(
            e,
            t,
            i,
            a
          ), t.child;
        case 6:
          return e === null && mt && (e = t.pendingProps, a = O(), i = a.ancestorInfo.current, e = i != null ? Mf(
            e,
            i.tag,
            a.ancestorInfo.implicitRootScope
          ) : !0, a = nl, (i = !a) || (i = Hl(
            a,
            t.pendingProps,
            Yi
          ), i !== null ? (t.stateNode = i, Na = t, nl = null, i = !0) : i = !1, i = !i), i && (e && Fh(t, a), xn(t))), null;
        case 13:
          return Fp(e, t, a);
        case 4:
          return Gt(
            t,
            t.stateNode.containerInfo
          ), i = t.pendingProps, e === null ? t.child = ph(
            t,
            null,
            i,
            a
          ) : al(
            e,
            t,
            i,
            a
          ), t.child;
        case 11:
          return Jr(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 7:
          return al(
            e,
            t,
            t.pendingProps,
            a
          ), t.child;
        case 8:
          return al(
            e,
            t,
            t.pendingProps.children,
            a
          ), t.child;
        case 12:
          return t.flags |= 4, t.flags |= 2048, i = t.stateNode, i.effectDuration = -0, i.passiveEffectDuration = -0, al(
            e,
            t,
            t.pendingProps.children,
            a
          ), t.child;
        case 10:
          return i = t.type, o = t.pendingProps, f = o.value, "value" in o || lb || (lb = !0, console.error(
            "The `value` prop is required for the `<Context.Provider>`. Did you misspell it or forget to pass it?"
          )), ci(t, i, f), al(
            e,
            t,
            o.children,
            a
          ), t.child;
        case 9:
          return o = t.type._context, i = t.pendingProps.children, typeof i != "function" && console.error(
            "A context consumer was rendered with multiple children, or a child that isn't a function. A context consumer expects a single child that is a function. If you did pass a function, make sure there is no trailing or leading whitespace around it."
          ), fi(t), o = Ct(o), wt(t), i = i0(
            i,
            o,
            void 0
          ), na(), t.flags |= 1, al(
            e,
            t,
            i,
            a
          ), t.child;
        case 14:
          return Xn(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 15:
          return ts(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 19:
          return Oy(
            e,
            t,
            a
          );
        case 31:
          return i = t.pendingProps, a = t.mode, i = {
            mode: i.mode,
            children: i.children
          }, e === null ? (e = ns(
            i,
            a
          ), e.ref = t.ref, t.child = e, e.return = t, t = e) : (e = Cn(e.child, i), e.ref = t.ref, t.child = e, e.return = t, t = e), t;
        case 22:
          return kr(e, t, a);
        case 24:
          return fi(t), i = Ct(jl), e === null ? (o = iy(), o === null && (o = xt, f = Bf(), o.pooledCache = f, cc(f), f !== null && (o.pooledCacheLanes |= a), o = f), t.memoizedState = {
            parent: i,
            cache: o
          }, ca(t), ci(t, jl, o)) : ((e.lanes & a) !== 0 && (ri(e, t), yo(t, null, null, a), qn()), o = e.memoizedState, f = t.memoizedState, o.parent !== i ? (o = {
            parent: i,
            cache: i
          }, t.memoizedState = o, t.lanes === 0 && (t.memoizedState = t.updateQueue.baseState = o), ci(t, jl, i)) : (i = f.cache, ci(t, jl, i), i !== o.cache && ay(
            t,
            [jl],
            a,
            !0
          ))), al(
            e,
            t,
            t.pendingProps.children,
            a
          ), t.child;
        case 29:
          throw t.pendingProps;
      }
      throw Error(
        "Unknown unit of work tag (" + t.tag + "). This error is likely caused by a bug in React. Please file an issue."
      );
    }
    function ra(e) {
      e.flags |= 4;
    }
    function us(e, t) {
      if (t.type !== "stylesheet" || (t.state.loading & pu) !== ar)
        e.flags &= -16777217;
      else if (e.flags |= 16777216, !bs(t)) {
        if (t = mu.current, t !== null && ((ut & 4194048) === ut ? Li !== null : (ut & 62914560) !== ut && (ut & 536870912) === 0 || t !== Li))
          throw Pm = Pg, d1;
        e.flags |= 8192;
      }
    }
    function is(e, t) {
      t !== null && (e.flags |= 4), e.flags & 16384 && (t = e.tag !== 22 ? Mn() : 536870912, e.lanes |= t, Fs |= t);
    }
    function Si(e, t) {
      if (!mt)
        switch (e.tailMode) {
          case "hidden":
            t = e.tail;
            for (var a = null; t !== null; )
              t.alternate !== null && (a = t), t = t.sibling;
            a === null ? e.tail = null : a.sibling = null;
            break;
          case "collapsed":
            a = e.tail;
            for (var i = null; a !== null; )
              a.alternate !== null && (i = a), a = a.sibling;
            i === null ? t || e.tail === null ? e.tail = null : e.tail.sibling = null : i.sibling = null;
        }
    }
    function Dt(e) {
      var t = e.alternate !== null && e.alternate.child === e.child, a = 0, i = 0;
      if (t)
        if ((e.mode & ta) !== Bt) {
          for (var o = e.selfBaseDuration, f = e.child; f !== null; )
            a |= f.lanes | f.childLanes, i |= f.subtreeFlags & 65011712, i |= f.flags & 65011712, o += f.treeBaseDuration, f = f.sibling;
          e.treeBaseDuration = o;
        } else
          for (o = e.child; o !== null; )
            a |= o.lanes | o.childLanes, i |= o.subtreeFlags & 65011712, i |= o.flags & 65011712, o.return = e, o = o.sibling;
      else if ((e.mode & ta) !== Bt) {
        o = e.actualDuration, f = e.selfBaseDuration;
        for (var d = e.child; d !== null; )
          a |= d.lanes | d.childLanes, i |= d.subtreeFlags, i |= d.flags, o += d.actualDuration, f += d.treeBaseDuration, d = d.sibling;
        e.actualDuration = o, e.treeBaseDuration = f;
      } else
        for (o = e.child; o !== null; )
          a |= o.lanes | o.childLanes, i |= o.subtreeFlags, i |= o.flags, o.return = e, o = o.sibling;
      return e.subtreeFlags |= i, e.childLanes = a, t;
    }
    function Ip(e, t, a) {
      var i = t.pendingProps;
      switch (Dr(t), t.tag) {
        case 31:
        case 16:
        case 15:
        case 0:
        case 11:
        case 7:
        case 8:
        case 12:
        case 9:
        case 14:
          return Dt(t), null;
        case 1:
          return Dt(t), null;
        case 3:
          return a = t.stateNode, i = null, e !== null && (i = e.memoizedState.cache), t.memoizedState.cache !== i && (t.flags |= 2048), Ru(jl, t), pt(t), a.pendingContext && (a.context = a.pendingContext, a.pendingContext = null), (e === null || e.child === null) && (uc(t) ? (ty(), ra(t)) : e === null || e.memoizedState.isDehydrated && (t.flags & 256) === 0 || (t.flags |= 1024, ey())), Dt(t), null;
        case 26:
          return a = t.memoizedState, e === null ? (ra(t), a !== null ? (Dt(t), us(
            t,
            a
          )) : (Dt(t), t.flags &= -16777217)) : a ? a !== e.memoizedState ? (ra(t), Dt(t), us(
            t,
            a
          )) : (Dt(t), t.flags &= -16777217) : (e.memoizedProps !== i && ra(t), Dt(t), t.flags &= -16777217), null;
        case 27:
          P(t), a = Ot(au.current);
          var o = t.type;
          if (e !== null && t.stateNode != null)
            e.memoizedProps !== i && ra(t);
          else {
            if (!i) {
              if (t.stateNode === null)
                throw Error(
                  "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
                );
              return Dt(t), null;
            }
            e = O(), uc(t) ? Ih(t) : (e = sm(
              o,
              i,
              a,
              e,
              !0
            ), t.stateNode = e, ra(t));
          }
          return Dt(t), null;
        case 5:
          if (P(t), a = t.type, e !== null && t.stateNode != null)
            e.memoizedProps !== i && ra(t);
          else {
            if (!i) {
              if (t.stateNode === null)
                throw Error(
                  "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
                );
              return Dt(t), null;
            }
            if (o = O(), uc(t))
              Ih(t);
            else {
              switch (e = Ot(au.current), mr(a, o.ancestorInfo), o = o.context, e = lt(e), o) {
                case Mh:
                  e = e.createElementNS(af, a);
                  break;
                case fg:
                  e = e.createElementNS(
                    Ys,
                    a
                  );
                  break;
                default:
                  switch (a) {
                    case "svg":
                      e = e.createElementNS(
                        af,
                        a
                      );
                      break;
                    case "math":
                      e = e.createElementNS(
                        Ys,
                        a
                      );
                      break;
                    case "script":
                      e = e.createElement("div"), e.innerHTML = "<script><\/script>", e = e.removeChild(e.firstChild);
                      break;
                    case "select":
                      e = typeof i.is == "string" ? e.createElement("select", { is: i.is }) : e.createElement("select"), i.multiple ? e.multiple = !0 : i.size && (e.size = i.size);
                      break;
                    default:
                      e = typeof i.is == "string" ? e.createElement(a, {
                        is: i.is
                      }) : e.createElement(a), a.indexOf("-") === -1 && (a !== a.toLowerCase() && console.error(
                        "<%s /> is using incorrect casing. Use PascalCase for React components, or lowercase for HTML elements.",
                        a
                      ), Object.prototype.toString.call(e) !== "[object HTMLUnknownElement]" || Lu.call(
                        Tb,
                        a
                      ) || (Tb[a] = !0, console.error(
                        "The tag <%s> is unrecognized in this browser. If you meant to render a React component, start its name with an uppercase letter.",
                        a
                      )));
                  }
              }
              e[Zl] = t, e[ga] = i;
              e: for (o = t.child; o !== null; ) {
                if (o.tag === 5 || o.tag === 6)
                  e.appendChild(o.stateNode);
                else if (o.tag !== 4 && o.tag !== 27 && o.child !== null) {
                  o.child.return = o, o = o.child;
                  continue;
                }
                if (o === t) break e;
                for (; o.sibling === null; ) {
                  if (o.return === null || o.return === t)
                    break e;
                  o = o.return;
                }
                o.sibling.return = o.return, o = o.sibling;
              }
              t.stateNode = e;
              e: switch (kt(e, a, i), a) {
                case "button":
                case "input":
                case "select":
                case "textarea":
                  e = !!i.autoFocus;
                  break e;
                case "img":
                  e = !0;
                  break e;
                default:
                  e = !1;
              }
              e && ra(t);
            }
          }
          return Dt(t), t.flags &= -16777217, null;
        case 6:
          if (e && t.stateNode != null)
            e.memoizedProps !== i && ra(t);
          else {
            if (typeof i != "string" && t.stateNode === null)
              throw Error(
                "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
              );
            if (e = Ot(au.current), a = O(), uc(t)) {
              e = t.stateNode, a = t.memoizedProps, o = !Vc, i = null;
              var f = Na;
              if (f !== null)
                switch (f.tag) {
                  case 3:
                    o && (o = Ud(
                      e,
                      a,
                      i
                    ), o !== null && (sn(t, 0).serverProps = o));
                    break;
                  case 27:
                  case 5:
                    i = f.memoizedProps, o && (o = Ud(
                      e,
                      a,
                      i
                    ), o !== null && (sn(
                      t,
                      0
                    ).serverProps = o));
                }
              e[Zl] = t, e = !!(e.nodeValue === a || i !== null && i.suppressHydrationWarning === !0 || em(e.nodeValue, a)), e || xn(t);
            } else
              o = a.ancestorInfo.current, o != null && Mf(
                i,
                o.tag,
                a.ancestorInfo.implicitRootScope
              ), e = lt(e).createTextNode(
                i
              ), e[Zl] = t, t.stateNode = e;
          }
          return Dt(t), null;
        case 13:
          if (i = t.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
            if (o = uc(t), i !== null && i.dehydrated !== null) {
              if (e === null) {
                if (!o)
                  throw Error(
                    "A dehydrated suspense component was completed without a hydrated node. This is probably a bug in React."
                  );
                if (o = t.memoizedState, o = o !== null ? o.dehydrated : null, !o)
                  throw Error(
                    "Expected to have a hydrated suspense instance. This error is likely caused by a bug in React. Please file an issue."
                  );
                o[Zl] = t, Dt(t), (t.mode & ta) !== Bt && i !== null && (o = t.child, o !== null && (t.treeBaseDuration -= o.treeBaseDuration));
              } else
                ty(), ic(), (t.flags & 128) === 0 && (t.memoizedState = null), t.flags |= 4, Dt(t), (t.mode & ta) !== Bt && i !== null && (o = t.child, o !== null && (t.treeBaseDuration -= o.treeBaseDuration));
              o = !1;
            } else
              o = ey(), e !== null && e.memoizedState !== null && (e.memoizedState.hydrationErrors = o), o = !0;
            if (!o)
              return t.flags & 256 ? (za(t), t) : (za(t), null);
          }
          return za(t), (t.flags & 128) !== 0 ? (t.lanes = a, (t.mode & ta) !== Bt && Nn(t), t) : (a = i !== null, e = e !== null && e.memoizedState !== null, a && (i = t.child, o = null, i.alternate !== null && i.alternate.memoizedState !== null && i.alternate.memoizedState.cachePool !== null && (o = i.alternate.memoizedState.cachePool.pool), f = null, i.memoizedState !== null && i.memoizedState.cachePool !== null && (f = i.memoizedState.cachePool.pool), f !== o && (i.flags |= 2048)), a !== e && a && (t.child.flags |= 8192), is(t, t.updateQueue), Dt(t), (t.mode & ta) !== Bt && a && (e = t.child, e !== null && (t.treeBaseDuration -= e.treeBaseDuration)), null);
        case 4:
          return pt(t), e === null && Iy(
            t.stateNode.containerInfo
          ), Dt(t), null;
        case 10:
          return Ru(t.type, t), Dt(t), null;
        case 19:
          if (Se(Yl, t), o = t.memoizedState, o === null) return Dt(t), null;
          if (i = (t.flags & 128) !== 0, f = o.rendering, f === null)
            if (i) Si(o, !1);
            else {
              if (ul !== Kc || e !== null && (e.flags & 128) !== 0)
                for (e = t.child; e !== null; ) {
                  if (f = Cu(e), f !== null) {
                    for (t.flags |= 128, Si(o, !1), e = f.updateQueue, t.updateQueue = e, is(t, e), t.subtreeFlags = 0, e = a, a = t.child; a !== null; )
                      $h(a, e), a = a.sibling;
                    return ze(
                      Yl,
                      Yl.current & vh | ap,
                      t
                    ), t.child;
                  }
                  e = e.sibling;
                }
              o.tail !== null && nu() > Fv && (t.flags |= 128, i = !0, Si(o, !1), t.lanes = 4194304);
            }
          else {
            if (!i)
              if (e = Cu(f), e !== null) {
                if (t.flags |= 128, i = !0, e = e.updateQueue, t.updateQueue = e, is(t, e), Si(o, !0), o.tail === null && o.tailMode === "hidden" && !f.alternate && !mt)
                  return Dt(t), null;
              } else
                2 * nu() - o.renderingStartTime > Fv && a !== 536870912 && (t.flags |= 128, i = !0, Si(o, !1), t.lanes = 4194304);
            o.isBackwards ? (f.sibling = t.child, t.child = f) : (e = o.last, e !== null ? e.sibling = f : t.child = f, o.last = f);
          }
          return o.tail !== null ? (e = o.tail, o.rendering = e, o.tail = e.sibling, o.renderingStartTime = nu(), e.sibling = null, a = Yl.current, a = i ? a & vh | ap : a & vh, ze(Yl, a, t), e) : (Dt(t), null);
        case 22:
        case 23:
          return za(t), yn(t), i = t.memoizedState !== null, e !== null ? e.memoizedState !== null !== i && (t.flags |= 8192) : i && (t.flags |= 8192), i ? (a & 536870912) !== 0 && (t.flags & 128) === 0 && (Dt(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : Dt(t), a = t.updateQueue, a !== null && is(t, a.retryQueue), a = null, e !== null && e.memoizedState !== null && e.memoizedState.cachePool !== null && (a = e.memoizedState.cachePool.pool), i = null, t.memoizedState !== null && t.memoizedState.cachePool !== null && (i = t.memoizedState.cachePool.pool), i !== a && (t.flags |= 2048), e !== null && Se(Zs, t), null;
        case 24:
          return a = null, e !== null && (a = e.memoizedState.cache), t.memoizedState.cache !== a && (t.flags |= 2048), Ru(jl, t), Dt(t), null;
        case 25:
          return null;
        case 30:
          return null;
      }
      throw Error(
        "Unknown unit of work tag (" + t.tag + "). This error is likely caused by a bug in React. Please file an issue."
      );
    }
    function Pp(e, t) {
      switch (Dr(t), t.tag) {
        case 1:
          return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & ta) !== Bt && Nn(t), t) : null;
        case 3:
          return Ru(jl, t), pt(t), e = t.flags, (e & 65536) !== 0 && (e & 128) === 0 ? (t.flags = e & -65537 | 128, t) : null;
        case 26:
        case 27:
        case 5:
          return P(t), null;
        case 13:
          if (za(t), e = t.memoizedState, e !== null && e.dehydrated !== null) {
            if (t.alternate === null)
              throw Error(
                "Threw in newly mounted dehydrated component. This is likely a bug in React. Please file an issue."
              );
            ic();
          }
          return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & ta) !== Bt && Nn(t), t) : null;
        case 19:
          return Se(Yl, t), null;
        case 4:
          return pt(t), null;
        case 10:
          return Ru(t.type, t), null;
        case 22:
        case 23:
          return za(t), yn(t), e !== null && Se(Zs, t), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & ta) !== Bt && Nn(t), t) : null;
        case 24:
          return Ru(jl, t), null;
        case 25:
          return null;
        default:
          return null;
      }
    }
    function Dy(e, t) {
      switch (Dr(t), t.tag) {
        case 3:
          Ru(jl, t), pt(t);
          break;
        case 26:
        case 27:
        case 5:
          P(t);
          break;
        case 4:
          pt(t);
          break;
        case 13:
          za(t);
          break;
        case 19:
          Se(Yl, t);
          break;
        case 10:
          Ru(t.type, t);
          break;
        case 22:
        case 23:
          za(t), yn(t), e !== null && Se(Zs, t);
          break;
        case 24:
          Ru(jl, t);
      }
    }
    function gn(e) {
      return (e.mode & ta) !== Bt;
    }
    function zy(e, t) {
      gn(e) ? (dn(), pc(t, e), Ga()) : pc(t, e);
    }
    function ud(e, t, a) {
      gn(e) ? (dn(), vc(
        a,
        e,
        t
      ), Ga()) : vc(
        a,
        e,
        t
      );
    }
    function pc(e, t) {
      try {
        var a = t.updateQueue, i = a !== null ? a.lastEffect : null;
        if (i !== null) {
          var o = i.next;
          a = o;
          do {
            if ((a.tag & e) === e && ((e & Bl) !== du ? fe !== null && typeof fe.markComponentPassiveEffectMountStarted == "function" && fe.markComponentPassiveEffectMountStarted(
              t
            ) : (e & la) !== du && fe !== null && typeof fe.markComponentLayoutEffectMountStarted == "function" && fe.markComponentLayoutEffectMountStarted(
              t
            ), i = void 0, (e & wa) !== du && (Dh = !0), i = ye(
              t,
              VS,
              a
            ), (e & wa) !== du && (Dh = !1), (e & Bl) !== du ? fe !== null && typeof fe.markComponentPassiveEffectMountStopped == "function" && fe.markComponentPassiveEffectMountStopped() : (e & la) !== du && fe !== null && typeof fe.markComponentLayoutEffectMountStopped == "function" && fe.markComponentLayoutEffectMountStopped(), i !== void 0 && typeof i != "function")) {
              var f = void 0;
              f = (a.tag & la) !== 0 ? "useLayoutEffect" : (a.tag & wa) !== 0 ? "useInsertionEffect" : "useEffect";
              var d = void 0;
              d = i === null ? " You returned null. If your effect does not require clean up, return undefined (or nothing)." : typeof i.then == "function" ? `

It looks like you wrote ` + f + `(async () => ...) or returned a Promise. Instead, write the async function inside your effect and call it immediately:

` + f + `(() => {
  async function fetchData() {
    // You can await here
    const response = await MyAPI.getData(someId);
    // ...
  }
  fetchData();
}, [someId]); // Or [] if effect doesn't need props or state

Learn more about data fetching with Hooks: https://react.dev/link/hooks-data-fetching` : " You returned: " + i, ye(
                t,
                function(h, v) {
                  console.error(
                    "%s must not return anything besides a function, which is used for clean-up.%s",
                    h,
                    v
                  );
                },
                f,
                d
              );
            }
            a = a.next;
          } while (a !== o);
        }
      } catch (h) {
        Me(t, t.return, h);
      }
    }
    function vc(e, t, a) {
      try {
        var i = t.updateQueue, o = i !== null ? i.lastEffect : null;
        if (o !== null) {
          var f = o.next;
          i = f;
          do {
            if ((i.tag & e) === e) {
              var d = i.inst, h = d.destroy;
              h !== void 0 && (d.destroy = void 0, (e & Bl) !== du ? fe !== null && typeof fe.markComponentPassiveEffectUnmountStarted == "function" && fe.markComponentPassiveEffectUnmountStarted(
                t
              ) : (e & la) !== du && fe !== null && typeof fe.markComponentLayoutEffectUnmountStarted == "function" && fe.markComponentLayoutEffectUnmountStarted(
                t
              ), (e & wa) !== du && (Dh = !0), o = t, ye(
                o,
                XS,
                o,
                a,
                h
              ), (e & wa) !== du && (Dh = !1), (e & Bl) !== du ? fe !== null && typeof fe.markComponentPassiveEffectUnmountStopped == "function" && fe.markComponentPassiveEffectUnmountStopped() : (e & la) !== du && fe !== null && typeof fe.markComponentLayoutEffectUnmountStopped == "function" && fe.markComponentLayoutEffectUnmountStopped());
            }
            i = i.next;
          } while (i !== f);
        }
      } catch (v) {
        Me(t, t.return, v);
      }
    }
    function My(e, t) {
      gn(e) ? (dn(), pc(t, e), Ga()) : pc(t, e);
    }
    function cs(e, t, a) {
      gn(e) ? (dn(), vc(
        a,
        e,
        t
      ), Ga()) : vc(
        a,
        e,
        t
      );
    }
    function _y(e) {
      var t = e.updateQueue;
      if (t !== null) {
        var a = e.stateNode;
        e.type.defaultProps || "ref" in e.memoizedProps || bh || (a.props !== e.memoizedProps && console.error(
          "Expected %s props to match memoized props before processing the update queue. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
          re(e) || "instance"
        ), a.state !== e.memoizedState && console.error(
          "Expected %s state to match memoized state before processing the update queue. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
          re(e) || "instance"
        ));
        try {
          ye(
            e,
            Jp,
            t,
            a
          );
        } catch (i) {
          Me(e, e.return, i);
        }
      }
    }
    function ev(e, t, a) {
      return e.getSnapshotBeforeUpdate(t, a);
    }
    function _g(e, t) {
      var a = t.memoizedProps, i = t.memoizedState;
      t = e.stateNode, e.type.defaultProps || "ref" in e.memoizedProps || bh || (t.props !== e.memoizedProps && console.error(
        "Expected %s props to match memoized props before getSnapshotBeforeUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
        re(e) || "instance"
      ), t.state !== e.memoizedState && console.error(
        "Expected %s state to match memoized state before getSnapshotBeforeUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
        re(e) || "instance"
      ));
      try {
        var o = bi(
          e.type,
          a,
          e.elementType === e.type
        ), f = ye(
          e,
          ev,
          t,
          o,
          i
        );
        a = ab, f !== void 0 || a.has(e.type) || (a.add(e.type), ye(e, function() {
          console.error(
            "%s.getSnapshotBeforeUpdate(): A snapshot value (or null) must be returned. You have returned undefined.",
            re(e)
          );
        })), t.__reactInternalSnapshotBeforeUpdate = f;
      } catch (d) {
        Me(e, e.return, d);
      }
    }
    function id(e, t, a) {
      a.props = bi(
        e.type,
        e.memoizedProps
      ), a.state = e.memoizedState, gn(e) ? (dn(), ye(
        e,
        U1,
        e,
        t,
        a
      ), Ga()) : ye(
        e,
        U1,
        e,
        t,
        a
      );
    }
    function tv(e) {
      var t = e.ref;
      if (t !== null) {
        switch (e.tag) {
          case 26:
          case 27:
          case 5:
            var a = e.stateNode;
            break;
          case 30:
            a = e.stateNode;
            break;
          default:
            a = e.stateNode;
        }
        if (typeof t == "function")
          if (gn(e))
            try {
              dn(), e.refCleanup = t(a);
            } finally {
              Ga();
            }
          else e.refCleanup = t(a);
        else
          typeof t == "string" ? console.error("String refs are no longer supported.") : t.hasOwnProperty("current") || console.error(
            "Unexpected ref object provided for %s. Use either a ref-setter function or React.createRef().",
            re(e)
          ), t.current = a;
      }
    }
    function Mo(e, t) {
      try {
        ye(e, tv, e);
      } catch (a) {
        Me(e, t, a);
      }
    }
    function ka(e, t) {
      var a = e.ref, i = e.refCleanup;
      if (a !== null)
        if (typeof i == "function")
          try {
            if (gn(e))
              try {
                dn(), ye(e, i);
              } finally {
                Ga(e);
              }
            else ye(e, i);
          } catch (o) {
            Me(e, t, o);
          } finally {
            e.refCleanup = null, e = e.alternate, e != null && (e.refCleanup = null);
          }
        else if (typeof a == "function")
          try {
            if (gn(e))
              try {
                dn(), ye(e, a, null);
              } finally {
                Ga(e);
              }
            else ye(e, a, null);
          } catch (o) {
            Me(e, t, o);
          }
        else a.current = null;
    }
    function Uy(e, t, a, i) {
      var o = e.memoizedProps, f = o.id, d = o.onCommit;
      o = o.onRender, t = t === null ? "mount" : "update", Yv && (t = "nested-update"), typeof o == "function" && o(
        f,
        t,
        e.actualDuration,
        e.treeBaseDuration,
        e.actualStartTime,
        a
      ), typeof d == "function" && d(
        e.memoizedProps.id,
        t,
        i,
        a
      );
    }
    function lv(e, t, a, i) {
      var o = e.memoizedProps;
      e = o.id, o = o.onPostCommit, t = t === null ? "mount" : "update", Yv && (t = "nested-update"), typeof o == "function" && o(
        e,
        t,
        i,
        a
      );
    }
    function av(e) {
      var t = e.type, a = e.memoizedProps, i = e.stateNode;
      try {
        ye(
          e,
          qu,
          i,
          t,
          a,
          e
        );
      } catch (o) {
        Me(e, e.return, o);
      }
    }
    function Cy(e, t, a) {
      try {
        ye(
          e,
          $t,
          e.stateNode,
          e.type,
          a,
          t,
          e
        );
      } catch (i) {
        Me(e, e.return, i);
      }
    }
    function xy(e) {
      return e.tag === 5 || e.tag === 3 || e.tag === 26 || e.tag === 27 && eu(e.type) || e.tag === 4;
    }
    function gc(e) {
      e: for (; ; ) {
        for (; e.sibling === null; ) {
          if (e.return === null || xy(e.return)) return null;
          e = e.return;
        }
        for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
          if (e.tag === 27 && eu(e.type) || e.flags & 2 || e.child === null || e.tag === 4) continue e;
          e.child.return = e, e = e.child;
        }
        if (!(e.flags & 2)) return e.stateNode;
      }
    }
    function os(e, t, a) {
      var i = e.tag;
      if (i === 5 || i === 6)
        e = e.stateNode, t ? (a.nodeType === 9 ? a.body : a.nodeName === "HTML" ? a.ownerDocument.body : a).insertBefore(e, t) : (t = a.nodeType === 9 ? a.body : a.nodeName === "HTML" ? a.ownerDocument.body : a, t.appendChild(e), a = a._reactRootContainer, a != null || t.onclick !== null || (t.onclick = wu));
      else if (i !== 4 && (i === 27 && eu(e.type) && (a = e.stateNode, t = null), e = e.child, e !== null))
        for (os(e, t, a), e = e.sibling; e !== null; )
          os(e, t, a), e = e.sibling;
    }
    function bc(e, t, a) {
      var i = e.tag;
      if (i === 5 || i === 6)
        e = e.stateNode, t ? a.insertBefore(e, t) : a.appendChild(e);
      else if (i !== 4 && (i === 27 && eu(e.type) && (a = e.stateNode), e = e.child, e !== null))
        for (bc(e, t, a), e = e.sibling; e !== null; )
          bc(e, t, a), e = e.sibling;
    }
    function nv(e) {
      for (var t, a = e.return; a !== null; ) {
        if (xy(a)) {
          t = a;
          break;
        }
        a = a.return;
      }
      if (t == null)
        throw Error(
          "Expected to find a host parent. This error is likely caused by a bug in React. Please file an issue."
        );
      switch (t.tag) {
        case 27:
          t = t.stateNode, a = gc(e), bc(
            e,
            a,
            t
          );
          break;
        case 5:
          a = t.stateNode, t.flags & 32 && (ju(a), t.flags &= -33), t = gc(e), bc(
            e,
            t,
            a
          );
          break;
        case 3:
        case 4:
          t = t.stateNode.containerInfo, a = gc(e), os(
            e,
            a,
            t
          );
          break;
        default:
          throw Error(
            "Invalid host parent fiber. This error is likely caused by a bug in React. Please file an issue."
          );
      }
    }
    function Hy(e) {
      var t = e.stateNode, a = e.memoizedProps;
      try {
        ye(
          e,
          Ua,
          e.type,
          a,
          t,
          e
        );
      } catch (i) {
        Me(e, e.return, i);
      }
    }
    function cd(e, t) {
      if (e = e.containerInfo, H0 = hg, e = Bp(e), Zh(e)) {
        if ("selectionStart" in e)
          var a = {
            start: e.selectionStart,
            end: e.selectionEnd
          };
        else
          e: {
            a = (a = e.ownerDocument) && a.defaultView || window;
            var i = a.getSelection && a.getSelection();
            if (i && i.rangeCount !== 0) {
              a = i.anchorNode;
              var o = i.anchorOffset, f = i.focusNode;
              i = i.focusOffset;
              try {
                a.nodeType, f.nodeType;
              } catch {
                a = null;
                break e;
              }
              var d = 0, h = -1, v = -1, b = 0, B = 0, X = e, N = null;
              t: for (; ; ) {
                for (var Q; X !== a || o !== 0 && X.nodeType !== 3 || (h = d + o), X !== f || i !== 0 && X.nodeType !== 3 || (v = d + i), X.nodeType === 3 && (d += X.nodeValue.length), (Q = X.firstChild) !== null; )
                  N = X, X = Q;
                for (; ; ) {
                  if (X === e) break t;
                  if (N === a && ++b === o && (h = d), N === f && ++B === i && (v = d), (Q = X.nextSibling) !== null) break;
                  X = N, N = X.parentNode;
                }
                X = Q;
              }
              a = h === -1 || v === -1 ? null : { start: h, end: v };
            } else a = null;
          }
        a = a || { start: 0, end: 0 };
      } else a = null;
      for (N0 = {
        focusedElem: e,
        selectionRange: a
      }, hg = !1, Jl = t; Jl !== null; )
        if (t = Jl, e = t.child, (t.subtreeFlags & 1024) !== 0 && e !== null)
          e.return = t, Jl = e;
        else
          for (; Jl !== null; ) {
            switch (e = t = Jl, a = e.alternate, o = e.flags, e.tag) {
              case 0:
                break;
              case 11:
              case 15:
                break;
              case 1:
                (o & 1024) !== 0 && a !== null && _g(e, a);
                break;
              case 3:
                if ((o & 1024) !== 0) {
                  if (e = e.stateNode.containerInfo, a = e.nodeType, a === 9)
                    Yo(e);
                  else if (a === 1)
                    switch (e.nodeName) {
                      case "HEAD":
                      case "HTML":
                      case "BODY":
                        Yo(e);
                        break;
                      default:
                        e.textContent = "";
                    }
                }
                break;
              case 5:
              case 26:
              case 27:
              case 6:
              case 4:
              case 17:
                break;
              default:
                if ((o & 1024) !== 0)
                  throw Error(
                    "This unit of work tag should not have side-effects. This error is likely caused by a bug in React. Please file an issue."
                  );
            }
            if (e = t.sibling, e !== null) {
              e.return = t.return, Jl = e;
              break;
            }
            Jl = t.return;
          }
    }
    function Ny(e, t, a) {
      var i = a.flags;
      switch (a.tag) {
        case 0:
        case 11:
        case 15:
          Zn(e, a), i & 4 && zy(a, la | hu);
          break;
        case 1:
          if (Zn(e, a), i & 4)
            if (e = a.stateNode, t === null)
              a.type.defaultProps || "ref" in a.memoizedProps || bh || (e.props !== a.memoizedProps && console.error(
                "Expected %s props to match memoized props before componentDidMount. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
                re(a) || "instance"
              ), e.state !== a.memoizedState && console.error(
                "Expected %s state to match memoized state before componentDidMount. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
                re(a) || "instance"
              )), gn(a) ? (dn(), ye(
                a,
                c0,
                a,
                e
              ), Ga()) : ye(
                a,
                c0,
                a,
                e
              );
            else {
              var o = bi(
                a.type,
                t.memoizedProps
              );
              t = t.memoizedState, a.type.defaultProps || "ref" in a.memoizedProps || bh || (e.props !== a.memoizedProps && console.error(
                "Expected %s props to match memoized props before componentDidUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
                re(a) || "instance"
              ), e.state !== a.memoizedState && console.error(
                "Expected %s state to match memoized state before componentDidUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
                re(a) || "instance"
              )), gn(a) ? (dn(), ye(
                a,
                z1,
                a,
                e,
                o,
                t,
                e.__reactInternalSnapshotBeforeUpdate
              ), Ga()) : ye(
                a,
                z1,
                a,
                e,
                o,
                t,
                e.__reactInternalSnapshotBeforeUpdate
              );
            }
          i & 64 && _y(a), i & 512 && Mo(a, a.return);
          break;
        case 3:
          if (t = rn(), Zn(e, a), i & 64 && (i = a.updateQueue, i !== null)) {
            if (o = null, a.child !== null)
              switch (a.child.tag) {
                case 27:
                case 5:
                  o = a.child.stateNode;
                  break;
                case 1:
                  o = a.child.stateNode;
              }
            try {
              ye(
                a,
                Jp,
                i,
                o
              );
            } catch (d) {
              Me(a, a.return, d);
            }
          }
          e.effectDuration += si(t);
          break;
        case 27:
          t === null && i & 4 && Hy(a);
        case 26:
        case 5:
          Zn(e, a), t === null && i & 4 && av(a), i & 512 && Mo(a, a.return);
          break;
        case 12:
          if (i & 4) {
            i = rn(), Zn(e, a), e = a.stateNode, e.effectDuration += oc(i);
            try {
              ye(
                a,
                Uy,
                a,
                t,
                Bv,
                e.effectDuration
              );
            } catch (d) {
              Me(a, a.return, d);
            }
          } else Zn(e, a);
          break;
        case 13:
          Zn(e, a), i & 4 && _o(e, a), i & 64 && (e = a.memoizedState, e !== null && (e = e.dehydrated, e !== null && (a = ps.bind(
            null,
            a
          ), Go(e, a))));
          break;
        case 22:
          if (i = a.memoizedState !== null || Zc, !i) {
            t = t !== null && t.memoizedState !== null || hl, o = Zc;
            var f = hl;
            Zc = i, (hl = t) && !f ? Kn(
              e,
              a,
              (a.subtreeFlags & 8772) !== 0
            ) : Zn(e, a), Zc = o, hl = f;
          }
          break;
        case 30:
          break;
        default:
          Zn(e, a);
      }
    }
    function wy(e) {
      var t = e.alternate;
      t !== null && (e.alternate = null, wy(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && nn(t)), e.stateNode = null, e._debugOwner = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
    }
    function xu(e, t, a) {
      for (a = a.child; a !== null; )
        Sc(
          e,
          t,
          a
        ), a = a.sibling;
    }
    function Sc(e, t, a) {
      if (wl && typeof wl.onCommitFiberUnmount == "function")
        try {
          wl.onCommitFiberUnmount(Hi, a);
        } catch (f) {
          va || (va = !0, console.error(
            "React instrumentation encountered an error: %s",
            f
          ));
        }
      switch (a.tag) {
        case 26:
          hl || ka(a, t), xu(
            e,
            t,
            a
          ), a.memoizedState ? a.memoizedState.count-- : a.stateNode && (a = a.stateNode, a.parentNode.removeChild(a));
          break;
        case 27:
          hl || ka(a, t);
          var i = Rl, o = ln;
          eu(a.type) && (Rl = a.stateNode, ln = !1), xu(
            e,
            t,
            a
          ), ye(
            a,
            Vo,
            a.stateNode
          ), Rl = i, ln = o;
          break;
        case 5:
          hl || ka(a, t);
        case 6:
          if (i = Rl, o = ln, Rl = null, xu(
            e,
            t,
            a
          ), Rl = i, ln = o, Rl !== null)
            if (ln)
              try {
                ye(
                  a,
                  jo,
                  Rl,
                  a.stateNode
                );
              } catch (f) {
                Me(
                  a,
                  t,
                  f
                );
              }
            else
              try {
                ye(
                  a,
                  Fa,
                  Rl,
                  a.stateNode
                );
              } catch (f) {
                Me(
                  a,
                  t,
                  f
                );
              }
          break;
        case 18:
          Rl !== null && (ln ? (e = Rl, Bo(
            e.nodeType === 9 ? e.body : e.nodeName === "HTML" ? e.ownerDocument.body : e,
            a.stateNode
          ), Hc(e)) : Bo(Rl, a.stateNode));
          break;
        case 4:
          i = Rl, o = ln, Rl = a.stateNode.containerInfo, ln = !0, xu(
            e,
            t,
            a
          ), Rl = i, ln = o;
          break;
        case 0:
        case 11:
        case 14:
        case 15:
          hl || vc(
            wa,
            a,
            t
          ), hl || ud(
            a,
            t,
            la
          ), xu(
            e,
            t,
            a
          );
          break;
        case 1:
          hl || (ka(a, t), i = a.stateNode, typeof i.componentWillUnmount == "function" && id(
            a,
            t,
            i
          )), xu(
            e,
            t,
            a
          );
          break;
        case 21:
          xu(
            e,
            t,
            a
          );
          break;
        case 22:
          hl = (i = hl) || a.memoizedState !== null, xu(
            e,
            t,
            a
          ), hl = i;
          break;
        default:
          xu(
            e,
            t,
            a
          );
      }
    }
    function _o(e, t) {
      if (t.memoizedState === null && (e = t.alternate, e !== null && (e = e.memoizedState, e !== null && (e = e.dehydrated, e !== null))))
        try {
          ye(
            t,
            _a,
            e
          );
        } catch (a) {
          Me(t, t.return, a);
        }
    }
    function od(e) {
      switch (e.tag) {
        case 13:
        case 19:
          var t = e.stateNode;
          return t === null && (t = e.stateNode = new nb()), t;
        case 22:
          return e = e.stateNode, t = e._retryCache, t === null && (t = e._retryCache = new nb()), t;
        default:
          throw Error(
            "Unexpected Suspense handler tag (" + e.tag + "). This is a bug in React."
          );
      }
    }
    function Tc(e, t) {
      var a = od(e);
      t.forEach(function(i) {
        var o = Ri.bind(null, e, i);
        if (!a.has(i)) {
          if (a.add(i), Ft)
            if (Sh !== null && Th !== null)
              wo(Th, Sh);
            else
              throw Error(
                "Expected finished root and lanes to be set. This is a bug in React."
              );
          i.then(o, o);
        }
      });
    }
    function Vl(e, t) {
      var a = t.deletions;
      if (a !== null)
        for (var i = 0; i < a.length; i++) {
          var o = e, f = t, d = a[i], h = f;
          e: for (; h !== null; ) {
            switch (h.tag) {
              case 27:
                if (eu(h.type)) {
                  Rl = h.stateNode, ln = !1;
                  break e;
                }
                break;
              case 5:
                Rl = h.stateNode, ln = !1;
                break e;
              case 3:
              case 4:
                Rl = h.stateNode.containerInfo, ln = !0;
                break e;
            }
            h = h.return;
          }
          if (Rl === null)
            throw Error(
              "Expected to find a host parent. This error is likely caused by a bug in React. Please file an issue."
            );
          Sc(o, f, d), Rl = null, ln = !1, o = d, f = o.alternate, f !== null && (f.return = null), o.return = null;
        }
      if (t.subtreeFlags & 13878)
        for (t = t.child; t !== null; )
          qy(t, e), t = t.sibling;
    }
    function qy(e, t) {
      var a = e.alternate, i = e.flags;
      switch (e.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          Vl(t, e), da(e), i & 4 && (vc(
            wa | hu,
            e,
            e.return
          ), pc(wa | hu, e), ud(
            e,
            e.return,
            la | hu
          ));
          break;
        case 1:
          Vl(t, e), da(e), i & 512 && (hl || a === null || ka(a, a.return)), i & 64 && Zc && (e = e.updateQueue, e !== null && (i = e.callbacks, i !== null && (a = e.shared.hiddenCallbacks, e.shared.hiddenCallbacks = a === null ? i : a.concat(i))));
          break;
        case 26:
          var o = $u;
          if (Vl(t, e), da(e), i & 512 && (hl || a === null || ka(a, a.return)), i & 4)
            if (t = a !== null ? a.memoizedState : null, i = e.memoizedState, a === null)
              if (i === null)
                if (e.stateNode === null) {
                  e: {
                    i = e.type, a = e.memoizedProps, t = o.ownerDocument || o;
                    t: switch (i) {
                      case "title":
                        o = t.getElementsByTagName("title")[0], (!o || o[Po] || o[Zl] || o.namespaceURI === af || o.hasAttribute("itemprop")) && (o = t.createElement(i), t.head.insertBefore(
                          o,
                          t.querySelector("head > title")
                        )), kt(o, i, a), o[Zl] = e, D(o), i = o;
                        break e;
                      case "link":
                        var f = ym(
                          "link",
                          "href",
                          t
                        ).get(i + (a.href || ""));
                        if (f) {
                          for (var d = 0; d < f.length; d++)
                            if (o = f[d], o.getAttribute("href") === (a.href == null || a.href === "" ? null : a.href) && o.getAttribute("rel") === (a.rel == null ? null : a.rel) && o.getAttribute("title") === (a.title == null ? null : a.title) && o.getAttribute("crossorigin") === (a.crossOrigin == null ? null : a.crossOrigin)) {
                              f.splice(d, 1);
                              break t;
                            }
                        }
                        o = t.createElement(i), kt(o, i, a), t.head.appendChild(o);
                        break;
                      case "meta":
                        if (f = ym(
                          "meta",
                          "content",
                          t
                        ).get(i + (a.content || ""))) {
                          for (d = 0; d < f.length; d++)
                            if (o = f[d], J(
                              a.content,
                              "content"
                            ), o.getAttribute("content") === (a.content == null ? null : "" + a.content) && o.getAttribute("name") === (a.name == null ? null : a.name) && o.getAttribute("property") === (a.property == null ? null : a.property) && o.getAttribute("http-equiv") === (a.httpEquiv == null ? null : a.httpEquiv) && o.getAttribute("charset") === (a.charSet == null ? null : a.charSet)) {
                              f.splice(d, 1);
                              break t;
                            }
                        }
                        o = t.createElement(i), kt(o, i, a), t.head.appendChild(o);
                        break;
                      default:
                        throw Error(
                          'getNodesForType encountered a type it did not expect: "' + i + '". This is a bug in React.'
                        );
                    }
                    o[Zl] = e, D(o), i = o;
                  }
                  e.stateNode = i;
                } else
                  mm(
                    o,
                    e.type,
                    e.stateNode
                  );
              else
                e.stateNode = Cd(
                  o,
                  i,
                  e.memoizedProps
                );
            else
              t !== i ? (t === null ? a.stateNode !== null && (a = a.stateNode, a.parentNode.removeChild(a)) : t.count--, i === null ? mm(
                o,
                e.type,
                e.stateNode
              ) : Cd(
                o,
                i,
                e.memoizedProps
              )) : i === null && e.stateNode !== null && Cy(
                e,
                e.memoizedProps,
                a.memoizedProps
              );
          break;
        case 27:
          Vl(t, e), da(e), i & 512 && (hl || a === null || ka(a, a.return)), a !== null && i & 4 && Cy(
            e,
            e.memoizedProps,
            a.memoizedProps
          );
          break;
        case 5:
          if (Vl(t, e), da(e), i & 512 && (hl || a === null || ka(a, a.return)), e.flags & 32) {
            t = e.stateNode;
            try {
              ye(e, ju, t);
            } catch (B) {
              Me(e, e.return, B);
            }
          }
          i & 4 && e.stateNode != null && (t = e.memoizedProps, Cy(
            e,
            t,
            a !== null ? a.memoizedProps : t
          )), i & 1024 && (y0 = !0, e.type !== "form" && console.error(
            "Unexpected host component type. Expected a form. This is a bug in React."
          ));
          break;
        case 6:
          if (Vl(t, e), da(e), i & 4) {
            if (e.stateNode === null)
              throw Error(
                "This should have a text node initialized. This error is likely caused by a bug in React. Please file an issue."
              );
            i = e.memoizedProps, a = a !== null ? a.memoizedProps : i, t = e.stateNode;
            try {
              ye(
                e,
                _c,
                t,
                a,
                i
              );
            } catch (B) {
              Me(e, e.return, B);
            }
          }
          break;
        case 3:
          if (o = rn(), sg = null, f = $u, $u = gs(t.containerInfo), Vl(t, e), $u = f, da(e), i & 4 && a !== null && a.memoizedState.isDehydrated)
            try {
              ye(
                e,
                fm,
                t.containerInfo
              );
            } catch (B) {
              Me(e, e.return, B);
            }
          y0 && (y0 = !1, Ec(e)), t.effectDuration += si(o);
          break;
        case 4:
          i = $u, $u = gs(
            e.stateNode.containerInfo
          ), Vl(t, e), da(e), $u = i;
          break;
        case 12:
          i = rn(), Vl(t, e), da(e), e.stateNode.effectDuration += oc(i);
          break;
        case 13:
          Vl(t, e), da(e), e.child.flags & 8192 && e.memoizedState !== null != (a !== null && a.memoizedState !== null) && (S0 = nu()), i & 4 && (i = e.updateQueue, i !== null && (e.updateQueue = null, Tc(e, i)));
          break;
        case 22:
          o = e.memoizedState !== null;
          var h = a !== null && a.memoizedState !== null, v = Zc, b = hl;
          if (Zc = v || o, hl = b || h, Vl(t, e), hl = b, Zc = v, da(e), i & 8192)
            e: for (t = e.stateNode, t._visibility = o ? t._visibility & ~Nv : t._visibility | Nv, o && (a === null || h || Zc || hl || Xl(e)), a = null, t = e; ; ) {
              if (t.tag === 5 || t.tag === 26) {
                if (a === null) {
                  h = a = t;
                  try {
                    f = h.stateNode, o ? ye(h, ma, f) : ye(
                      h,
                      cm,
                      h.stateNode,
                      h.memoizedProps
                    );
                  } catch (B) {
                    Me(h, h.return, B);
                  }
                }
              } else if (t.tag === 6) {
                if (a === null) {
                  h = t;
                  try {
                    d = h.stateNode, o ? ye(h, im, d) : ye(
                      h,
                      Md,
                      d,
                      h.memoizedProps
                    );
                  } catch (B) {
                    Me(h, h.return, B);
                  }
                }
              } else if ((t.tag !== 22 && t.tag !== 23 || t.memoizedState === null || t === e) && t.child !== null) {
                t.child.return = t, t = t.child;
                continue;
              }
              if (t === e) break e;
              for (; t.sibling === null; ) {
                if (t.return === null || t.return === e)
                  break e;
                a === t && (a = null), t = t.return;
              }
              a === t && (a = null), t.sibling.return = t.return, t = t.sibling;
            }
          i & 4 && (i = e.updateQueue, i !== null && (a = i.retryQueue, a !== null && (i.retryQueue = null, Tc(e, a))));
          break;
        case 19:
          Vl(t, e), da(e), i & 4 && (i = e.updateQueue, i !== null && (e.updateQueue = null, Tc(e, i)));
          break;
        case 30:
          break;
        case 21:
          break;
        default:
          Vl(t, e), da(e);
      }
    }
    function da(e) {
      var t = e.flags;
      if (t & 2) {
        try {
          ye(e, nv, e);
        } catch (a) {
          Me(e, e.return, a);
        }
        e.flags &= -3;
      }
      t & 4096 && (e.flags &= -4097);
    }
    function Ec(e) {
      if (e.subtreeFlags & 1024)
        for (e = e.child; e !== null; ) {
          var t = e;
          Ec(t), t.tag === 5 && t.flags & 1024 && t.stateNode.reset(), e = e.sibling;
        }
    }
    function Zn(e, t) {
      if (t.subtreeFlags & 8772)
        for (t = t.child; t !== null; )
          Ny(e, t.alternate, t), t = t.sibling;
    }
    function Ma(e) {
      switch (e.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          ud(
            e,
            e.return,
            la
          ), Xl(e);
          break;
        case 1:
          ka(e, e.return);
          var t = e.stateNode;
          typeof t.componentWillUnmount == "function" && id(
            e,
            e.return,
            t
          ), Xl(e);
          break;
        case 27:
          ye(
            e,
            Vo,
            e.stateNode
          );
        case 26:
        case 5:
          ka(e, e.return), Xl(e);
          break;
        case 22:
          e.memoizedState === null && Xl(e);
          break;
        case 30:
          Xl(e);
          break;
        default:
          Xl(e);
      }
    }
    function Xl(e) {
      for (e = e.child; e !== null; )
        Ma(e), e = e.sibling;
    }
    function Hu(e, t, a, i) {
      var o = a.flags;
      switch (a.tag) {
        case 0:
        case 11:
        case 15:
          Kn(
            e,
            a,
            i
          ), zy(a, la);
          break;
        case 1:
          if (Kn(
            e,
            a,
            i
          ), t = a.stateNode, typeof t.componentDidMount == "function" && ye(
            a,
            c0,
            a,
            t
          ), t = a.updateQueue, t !== null) {
            e = a.stateNode;
            try {
              ye(
                a,
                mo,
                t,
                e
              );
            } catch (f) {
              Me(a, a.return, f);
            }
          }
          i && o & 64 && _y(a), Mo(a, a.return);
          break;
        case 27:
          Hy(a);
        case 26:
        case 5:
          Kn(
            e,
            a,
            i
          ), i && t === null && o & 4 && av(a), Mo(a, a.return);
          break;
        case 12:
          if (i && o & 4) {
            o = rn(), Kn(
              e,
              a,
              i
            ), i = a.stateNode, i.effectDuration += oc(o);
            try {
              ye(
                a,
                Uy,
                a,
                t,
                Bv,
                i.effectDuration
              );
            } catch (f) {
              Me(a, a.return, f);
            }
          } else
            Kn(
              e,
              a,
              i
            );
          break;
        case 13:
          Kn(
            e,
            a,
            i
          ), i && o & 4 && _o(e, a);
          break;
        case 22:
          a.memoizedState === null && Kn(
            e,
            a,
            i
          ), Mo(a, a.return);
          break;
        case 30:
          break;
        default:
          Kn(
            e,
            a,
            i
          );
      }
    }
    function Kn(e, t, a) {
      for (a = a && (t.subtreeFlags & 8772) !== 0, t = t.child; t !== null; )
        Hu(
          e,
          t.alternate,
          t,
          a
        ), t = t.sibling;
    }
    function Jn(e, t) {
      var a = null;
      e !== null && e.memoizedState !== null && e.memoizedState.cachePool !== null && (a = e.memoizedState.cachePool.pool), e = null, t.memoizedState !== null && t.memoizedState.cachePool !== null && (e = t.memoizedState.cachePool.pool), e !== a && (e != null && cc(e), a != null && Hn(a));
    }
    function bn(e, t) {
      e = null, t.alternate !== null && (e = t.alternate.memoizedState.cache), t = t.memoizedState.cache, t !== e && (cc(t), e != null && Hn(e));
    }
    function zt(e, t, a, i) {
      if (t.subtreeFlags & 10256)
        for (t = t.child; t !== null; )
          fs(
            e,
            t,
            a,
            i
          ), t = t.sibling;
    }
    function fs(e, t, a, i) {
      var o = t.flags;
      switch (t.tag) {
        case 0:
        case 11:
        case 15:
          zt(
            e,
            t,
            a,
            i
          ), o & 2048 && My(t, Bl | hu);
          break;
        case 1:
          zt(
            e,
            t,
            a,
            i
          );
          break;
        case 3:
          var f = rn();
          zt(
            e,
            t,
            a,
            i
          ), o & 2048 && (a = null, t.alternate !== null && (a = t.alternate.memoizedState.cache), t = t.memoizedState.cache, t !== a && (cc(t), a != null && Hn(a))), e.passiveEffectDuration += si(f);
          break;
        case 12:
          if (o & 2048) {
            o = rn(), zt(
              e,
              t,
              a,
              i
            ), e = t.stateNode, e.passiveEffectDuration += oc(o);
            try {
              ye(
                t,
                lv,
                t,
                t.alternate,
                Bv,
                e.passiveEffectDuration
              );
            } catch (h) {
              Me(t, t.return, h);
            }
          } else
            zt(
              e,
              t,
              a,
              i
            );
          break;
        case 13:
          zt(
            e,
            t,
            a,
            i
          );
          break;
        case 23:
          break;
        case 22:
          f = t.stateNode;
          var d = t.alternate;
          t.memoizedState !== null ? f._visibility & Yc ? zt(
            e,
            t,
            a,
            i
          ) : Uo(
            e,
            t
          ) : f._visibility & Yc ? zt(
            e,
            t,
            a,
            i
          ) : (f._visibility |= Yc, Ti(
            e,
            t,
            a,
            i,
            (t.subtreeFlags & 10256) !== 0
          )), o & 2048 && Jn(d, t);
          break;
        case 24:
          zt(
            e,
            t,
            a,
            i
          ), o & 2048 && bn(t.alternate, t);
          break;
        default:
          zt(
            e,
            t,
            a,
            i
          );
      }
    }
    function Ti(e, t, a, i, o) {
      for (o = o && (t.subtreeFlags & 10256) !== 0, t = t.child; t !== null; )
        fd(
          e,
          t,
          a,
          i,
          o
        ), t = t.sibling;
    }
    function fd(e, t, a, i, o) {
      var f = t.flags;
      switch (t.tag) {
        case 0:
        case 11:
        case 15:
          Ti(
            e,
            t,
            a,
            i,
            o
          ), My(t, Bl);
          break;
        case 23:
          break;
        case 22:
          var d = t.stateNode;
          t.memoizedState !== null ? d._visibility & Yc ? Ti(
            e,
            t,
            a,
            i,
            o
          ) : Uo(
            e,
            t
          ) : (d._visibility |= Yc, Ti(
            e,
            t,
            a,
            i,
            o
          )), o && f & 2048 && Jn(
            t.alternate,
            t
          );
          break;
        case 24:
          Ti(
            e,
            t,
            a,
            i,
            o
          ), o && f & 2048 && bn(t.alternate, t);
          break;
        default:
          Ti(
            e,
            t,
            a,
            i,
            o
          );
      }
    }
    function Uo(e, t) {
      if (t.subtreeFlags & 10256)
        for (t = t.child; t !== null; ) {
          var a = e, i = t, o = i.flags;
          switch (i.tag) {
            case 22:
              Uo(
                a,
                i
              ), o & 2048 && Jn(
                i.alternate,
                i
              );
              break;
            case 24:
              Uo(
                a,
                i
              ), o & 2048 && bn(
                i.alternate,
                i
              );
              break;
            default:
              Uo(
                a,
                i
              );
          }
          t = t.sibling;
        }
    }
    function Ac(e) {
      if (e.subtreeFlags & np)
        for (e = e.child; e !== null; )
          Ei(e), e = e.sibling;
    }
    function Ei(e) {
      switch (e.tag) {
        case 26:
          Ac(e), e.flags & np && e.memoizedState !== null && mv(
            $u,
            e.memoizedState,
            e.memoizedProps
          );
          break;
        case 5:
          Ac(e);
          break;
        case 3:
        case 4:
          var t = $u;
          $u = gs(
            e.stateNode.containerInfo
          ), Ac(e), $u = t;
          break;
        case 22:
          e.memoizedState === null && (t = e.alternate, t !== null && t.memoizedState !== null ? (t = np, np = 16777216, Ac(e), np = t) : Ac(e));
          break;
        default:
          Ac(e);
      }
    }
    function ss(e) {
      var t = e.alternate;
      if (t !== null && (e = t.child, e !== null)) {
        t.child = null;
        do
          t = e.sibling, e.sibling = null, e = t;
        while (e !== null);
      }
    }
    function Co(e) {
      var t = e.deletions;
      if ((e.flags & 16) !== 0) {
        if (t !== null)
          for (var a = 0; a < t.length; a++) {
            var i = t[a];
            Jl = i, By(
              i,
              e
            );
          }
        ss(e);
      }
      if (e.subtreeFlags & 10256)
        for (e = e.child; e !== null; )
          jy(e), e = e.sibling;
    }
    function jy(e) {
      switch (e.tag) {
        case 0:
        case 11:
        case 15:
          Co(e), e.flags & 2048 && cs(
            e,
            e.return,
            Bl | hu
          );
          break;
        case 3:
          var t = rn();
          Co(e), e.stateNode.passiveEffectDuration += si(t);
          break;
        case 12:
          t = rn(), Co(e), e.stateNode.passiveEffectDuration += oc(t);
          break;
        case 22:
          t = e.stateNode, e.memoizedState !== null && t._visibility & Yc && (e.return === null || e.return.tag !== 13) ? (t._visibility &= ~Yc, rs(e)) : Co(e);
          break;
        default:
          Co(e);
      }
    }
    function rs(e) {
      var t = e.deletions;
      if ((e.flags & 16) !== 0) {
        if (t !== null)
          for (var a = 0; a < t.length; a++) {
            var i = t[a];
            Jl = i, By(
              i,
              e
            );
          }
        ss(e);
      }
      for (e = e.child; e !== null; )
        ds(e), e = e.sibling;
    }
    function ds(e) {
      switch (e.tag) {
        case 0:
        case 11:
        case 15:
          cs(
            e,
            e.return,
            Bl
          ), rs(e);
          break;
        case 22:
          var t = e.stateNode;
          t._visibility & Yc && (t._visibility &= ~Yc, rs(e));
          break;
        default:
          rs(e);
      }
    }
    function By(e, t) {
      for (; Jl !== null; ) {
        var a = Jl, i = a;
        switch (i.tag) {
          case 0:
          case 11:
          case 15:
            cs(
              i,
              t,
              Bl
            );
            break;
          case 23:
          case 22:
            i.memoizedState !== null && i.memoizedState.cachePool !== null && (i = i.memoizedState.cachePool.pool, i != null && cc(i));
            break;
          case 24:
            Hn(i.memoizedState.cache);
        }
        if (i = a.child, i !== null) i.return = a, Jl = i;
        else
          e: for (a = e; Jl !== null; ) {
            i = Jl;
            var o = i.sibling, f = i.return;
            if (wy(i), i === a) {
              Jl = null;
              break e;
            }
            if (o !== null) {
              o.return = f, Jl = o;
              break e;
            }
            Jl = f;
          }
      }
    }
    function Yy() {
      ZS.forEach(function(e) {
        return e();
      });
    }
    function Gy() {
      var e = typeof IS_REACT_ACT_ENVIRONMENT < "u" ? IS_REACT_ACT_ENVIRONMENT : void 0;
      return e || L.actQueue === null || console.error(
        "The current testing environment is not configured to support act(...)"
      ), e;
    }
    function ha(e) {
      if ((Et & qa) !== An && ut !== 0)
        return ut & -ut;
      var t = L.T;
      return t !== null ? (t._updatedFibers || (t._updatedFibers = /* @__PURE__ */ new Set()), t._updatedFibers.add(e), e = Qs, e !== 0 ? e : $y()) : Tf();
    }
    function uv() {
      On === 0 && (On = (ut & 536870912) === 0 || mt ? Ke() : 536870912);
      var e = mu.current;
      return e !== null && (e.flags |= 32), On;
    }
    function Kt(e, t, a) {
      if (Dh && console.error("useInsertionEffect must not schedule updates."), O0 && (Iv = !0), (e === xt && (Mt === $s || Mt === Ws) || e.cancelPendingCommit !== null) && (Oc(e, 0), Nu(
        e,
        ut,
        On,
        !1
      )), gu(e, a), (Et & qa) !== 0 && e === xt) {
        if (ba)
          switch (t.tag) {
            case 0:
            case 11:
            case 15:
              e = at && re(at) || "Unknown", yb.has(e) || (yb.add(e), t = re(t) || "Unknown", console.error(
                "Cannot update a component (`%s`) while rendering a different component (`%s`). To locate the bad setState() call inside `%s`, follow the stack trace as described in https://react.dev/link/setstate-in-render",
                t,
                e,
                e
              ));
              break;
            case 1:
              hb || (console.error(
                "Cannot update during an existing state transition (such as within `render`). Render methods should be a pure function of props and state."
              ), hb = !0);
          }
      } else
        Ft && Ba(e, t, a), fv(t), e === xt && ((Et & qa) === An && (rf |= a), ul === ks && Nu(
          e,
          ut,
          On,
          !1
        )), $a(e);
    }
    function rl(e, t, a) {
      if ((Et & (qa | Wu)) !== An)
        throw Error("Should not already be working.");
      var i = !a && (t & 124) === 0 && (t & e.expiredLanes) === 0 || Iu(e, t), o = i ? Vy(e, t) : hd(e, t, !0), f = i;
      do {
        if (o === Kc) {
          Rh && !i && Nu(e, t, 0, !1);
          break;
        } else {
          if (a = e.current.alternate, f && !iv(a)) {
            o = hd(e, t, !1), f = !1;
            continue;
          }
          if (o === Eh) {
            if (f = t, e.errorRecoveryDisabledLanes & f)
              var d = 0;
            else
              d = e.pendingLanes & -536870913, d = d !== 0 ? d : d & 536870912 ? 536870912 : 0;
            if (d !== 0) {
              t = d;
              e: {
                o = e;
                var h = d;
                d = sp;
                var v = o.current.memoizedState.isDehydrated;
                if (v && (Oc(
                  o,
                  h
                ).flags |= 256), h = hd(
                  o,
                  h,
                  !1
                ), h !== Eh) {
                  if (g0 && !v) {
                    o.errorRecoveryDisabledLanes |= f, rf |= f, o = ks;
                    break e;
                  }
                  o = ja, ja = d, o !== null && (ja === null ? ja = o : ja.push.apply(
                    ja,
                    o
                  ));
                }
                o = h;
              }
              if (f = !1, o !== Eh) continue;
            }
          }
          if (o === ip) {
            Oc(e, 0), Nu(e, t, 0, !0);
            break;
          }
          e: {
            switch (i = e, o) {
              case Kc:
              case ip:
                throw Error("Root did not complete. This is a bug in React.");
              case ks:
                if ((t & 4194048) !== t) break;
              case $v:
                Nu(
                  i,
                  t,
                  On,
                  !ff
                );
                break e;
              case Eh:
                ja = null;
                break;
              case m0:
              case ub:
                break;
              default:
                throw Error("Unknown root exit status.");
            }
            if (L.actQueue !== null)
              gd(
                i,
                a,
                t,
                ja,
                rp,
                Wv,
                On,
                rf,
                Fs
              );
            else {
              if ((t & 62914560) === t && (f = S0 + cb - nu(), 10 < f)) {
                if (Nu(
                  i,
                  t,
                  On,
                  !ff
                ), pl(i, 0, !0) !== 0) break e;
                i.timeoutHandle = Eb(
                  Sl.bind(
                    null,
                    i,
                    a,
                    ja,
                    rp,
                    Wv,
                    t,
                    On,
                    rf,
                    Fs,
                    ff,
                    o,
                    $S,
                    f1,
                    0
                  ),
                  f
                );
                break e;
              }
              Sl(
                i,
                a,
                ja,
                rp,
                Wv,
                t,
                On,
                rf,
                Fs,
                ff,
                o,
                JS,
                f1,
                0
              );
            }
          }
        }
        break;
      } while (!0);
      $a(e);
    }
    function Sl(e, t, a, i, o, f, d, h, v, b, B, X, N, Q) {
      if (e.timeoutHandle = lr, X = t.subtreeFlags, (X & 8192 || (X & 16785408) === 16785408) && (vp = { stylesheets: null, count: 0, unsuspend: yv }, Ei(t), X = pv(), X !== null)) {
        e.cancelPendingCommit = X(
          gd.bind(
            null,
            e,
            t,
            f,
            a,
            i,
            o,
            d,
            h,
            v,
            B,
            kS,
            N,
            Q
          )
        ), Nu(
          e,
          f,
          d,
          !b
        );
        return;
      }
      gd(
        e,
        t,
        f,
        a,
        i,
        o,
        d,
        h,
        v
      );
    }
    function iv(e) {
      for (var t = e; ; ) {
        var a = t.tag;
        if ((a === 0 || a === 11 || a === 15) && t.flags & 16384 && (a = t.updateQueue, a !== null && (a = a.stores, a !== null)))
          for (var i = 0; i < a.length; i++) {
            var o = a[i], f = o.getSnapshot;
            o = o.value;
            try {
              if (!Ha(f(), o)) return !1;
            } catch {
              return !1;
            }
          }
        if (a = t.child, t.subtreeFlags & 16384 && a !== null)
          a.return = t, t = a;
        else {
          if (t === e) break;
          for (; t.sibling === null; ) {
            if (t.return === null || t.return === e) return !0;
            t = t.return;
          }
          t.sibling.return = t.return, t = t.sibling;
        }
      }
      return !0;
    }
    function Nu(e, t, a, i) {
      t &= ~b0, t &= ~rf, e.suspendedLanes |= t, e.pingedLanes &= ~t, i && (e.warmLanes |= t), i = e.expirationTimes;
      for (var o = t; 0 < o; ) {
        var f = 31 - Ql(o), d = 1 << f;
        i[f] = -1, o &= ~d;
      }
      a !== 0 && Sf(e, a, t);
    }
    function Rc() {
      return (Et & (qa | Wu)) === An ? (Dc(0), !1) : !0;
    }
    function sd() {
      if (at !== null) {
        if (Mt === an)
          var e = at.return;
        else
          e = at, zr(), mn(e), mh = null, lp = 0, e = at;
        for (; e !== null; )
          Dy(e.alternate, e), e = e.return;
        at = null;
      }
    }
    function Oc(e, t) {
      var a = e.timeoutHandle;
      a !== lr && (e.timeoutHandle = lr, oT(a)), a = e.cancelPendingCommit, a !== null && (e.cancelPendingCommit = null, a()), sd(), xt = e, at = a = Cn(e.current, null), ut = t, Mt = an, Rn = null, ff = !1, Rh = Iu(e, t), g0 = !1, ul = Kc, Fs = On = b0 = rf = sf = 0, ja = sp = null, Wv = !1, (t & 8) !== 0 && (t |= t & 32);
      var i = e.entangledLanes;
      if (i !== 0)
        for (e = e.entanglements, i &= t; 0 < i; ) {
          var o = 31 - Ql(i), f = 1 << o;
          t |= e[o], i &= ~f;
        }
      return Vi = t, Hf(), t = c1(), 1e3 < t - i1 && (L.recentlyCreatedOwnerStacks = 0, i1 = t), Ju.discardPendingWarnings(), a;
    }
    function hs(e, t) {
      qe = null, L.H = Jv, L.getCurrentStack = null, ba = !1, xa = null, t === Im || t === Vv ? (t = fy(), Mt = op) : t === d1 ? (t = fy(), Mt = ib) : Mt = t === W1 ? v0 : t !== null && typeof t == "object" && typeof t.then == "function" ? Ah : cp, Rn = t;
      var a = at;
      if (a === null)
        ul = ip, zo(
          e,
          Ra(t, e.current)
        );
      else
        switch (a.mode & ta && Ou(a), na(), Mt) {
          case cp:
            fe !== null && typeof fe.markComponentErrored == "function" && fe.markComponentErrored(
              a,
              t,
              ut
            );
            break;
          case $s:
          case Ws:
          case op:
          case Ah:
          case fp:
            fe !== null && typeof fe.markComponentSuspended == "function" && fe.markComponentSuspended(
              a,
              t,
              ut
            );
        }
    }
    function rd() {
      var e = L.H;
      return L.H = Jv, e === null ? Jv : e;
    }
    function Ly() {
      var e = L.A;
      return L.A = QS, e;
    }
    function dd() {
      ul = ks, ff || (ut & 4194048) !== ut && mu.current !== null || (Rh = !0), (sf & 134217727) === 0 && (rf & 134217727) === 0 || xt === null || Nu(
        xt,
        ut,
        On,
        !1
      );
    }
    function hd(e, t, a) {
      var i = Et;
      Et |= qa;
      var o = rd(), f = Ly();
      if (xt !== e || ut !== t) {
        if (Ft) {
          var d = e.memoizedUpdaters;
          0 < d.size && (wo(e, ut), d.clear()), Dl(e, t);
        }
        rp = null, Oc(e, t);
      }
      Dn(t), t = !1, d = ul;
      e: do
        try {
          if (Mt !== an && at !== null) {
            var h = at, v = Rn;
            switch (Mt) {
              case v0:
                sd(), d = $v;
                break e;
              case op:
              case $s:
              case Ws:
              case Ah:
                mu.current === null && (t = !0);
                var b = Mt;
                if (Mt = an, Rn = null, Ai(e, h, v, b), a && Rh) {
                  d = Kc;
                  break e;
                }
                break;
              default:
                b = Mt, Mt = an, Rn = null, Ai(e, h, v, b);
            }
          }
          yd(), d = ul;
          break;
        } catch (B) {
          hs(e, B);
        }
      while (!0);
      return t && e.shellSuspendCounter++, zr(), Et = i, L.H = o, L.A = f, Zi(), at === null && (xt = null, ut = 0, Hf()), d;
    }
    function yd() {
      for (; at !== null; ) Qy(at);
    }
    function Vy(e, t) {
      var a = Et;
      Et |= qa;
      var i = rd(), o = Ly();
      if (xt !== e || ut !== t) {
        if (Ft) {
          var f = e.memoizedUpdaters;
          0 < f.size && (wo(e, ut), f.clear()), Dl(e, t);
        }
        rp = null, Fv = nu() + ob, Oc(e, t);
      } else
        Rh = Iu(
          e,
          t
        );
      Dn(t);
      e: do
        try {
          if (Mt !== an && at !== null)
            t: switch (t = at, f = Rn, Mt) {
              case cp:
                Mt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  cp
                );
                break;
              case $s:
              case Ws:
                if (oy(f)) {
                  Mt = an, Rn = null, md(t);
                  break;
                }
                t = function() {
                  Mt !== $s && Mt !== Ws || xt !== e || (Mt = fp), $a(e);
                }, f.then(t, t);
                break e;
              case op:
                Mt = fp;
                break e;
              case ib:
                Mt = p0;
                break e;
              case fp:
                oy(f) ? (Mt = an, Rn = null, md(t)) : (Mt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  fp
                ));
                break;
              case p0:
                var d = null;
                switch (at.tag) {
                  case 26:
                    d = at.memoizedState;
                  case 5:
                  case 27:
                    var h = at;
                    if (!d || bs(d)) {
                      Mt = an, Rn = null;
                      var v = h.sibling;
                      if (v !== null) at = v;
                      else {
                        var b = h.return;
                        b !== null ? (at = b, ys(b)) : at = null;
                      }
                      break t;
                    }
                    break;
                  default:
                    console.error(
                      "Unexpected type of fiber triggered a suspensey commit. This is a bug in React."
                    );
                }
                Mt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  p0
                );
                break;
              case Ah:
                Mt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  Ah
                );
                break;
              case v0:
                sd(), ul = $v;
                break e;
              default:
                throw Error(
                  "Unexpected SuspendedReason. This is a bug in React."
                );
            }
          L.actQueue !== null ? yd() : Xy();
          break;
        } catch (B) {
          hs(e, B);
        }
      while (!0);
      return zr(), L.H = i, L.A = o, Et = a, at !== null ? (fe !== null && typeof fe.markRenderYielded == "function" && fe.markRenderYielded(), Kc) : (Zi(), xt = null, ut = 0, Hf(), ul);
    }
    function Xy() {
      for (; at !== null && !Av(); )
        Qy(at);
    }
    function Qy(e) {
      var t = e.alternate;
      (e.mode & ta) !== Bt ? (Mr(e), t = ye(
        e,
        nd,
        t,
        e,
        Vi
      ), Ou(e)) : t = ye(
        e,
        nd,
        t,
        e,
        Vi
      ), e.memoizedProps = e.pendingProps, t === null ? ys(e) : at = t;
    }
    function md(e) {
      var t = ye(e, pd, e);
      e.memoizedProps = e.pendingProps, t === null ? ys(e) : at = t;
    }
    function pd(e) {
      var t = e.alternate, a = (e.mode & ta) !== Bt;
      switch (a && Mr(e), e.tag) {
        case 15:
        case 0:
          t = Ty(
            t,
            e,
            e.pendingProps,
            e.type,
            void 0,
            ut
          );
          break;
        case 11:
          t = Ty(
            t,
            e,
            e.pendingProps,
            e.type.render,
            e.ref,
            ut
          );
          break;
        case 5:
          mn(e);
        default:
          Dy(t, e), e = at = $h(e, Vi), t = nd(t, e, Vi);
      }
      return a && Ou(e), t;
    }
    function Ai(e, t, a, i) {
      zr(), mn(t), mh = null, lp = 0;
      var o = t.return;
      try {
        if (es(
          e,
          o,
          t,
          a,
          ut
        )) {
          ul = ip, zo(
            e,
            Ra(a, e.current)
          ), at = null;
          return;
        }
      } catch (f) {
        if (o !== null) throw at = o, f;
        ul = ip, zo(
          e,
          Ra(a, e.current)
        ), at = null;
        return;
      }
      t.flags & 32768 ? (mt || i === cp ? e = !0 : Rh || (ut & 536870912) !== 0 ? e = !1 : (ff = e = !0, (i === $s || i === Ws || i === op || i === Ah) && (i = mu.current, i !== null && i.tag === 13 && (i.flags |= 16384))), vd(t, e)) : ys(t);
    }
    function ys(e) {
      var t = e;
      do {
        if ((t.flags & 32768) !== 0) {
          vd(
            t,
            ff
          );
          return;
        }
        var a = t.alternate;
        if (e = t.return, Mr(t), a = ye(
          t,
          Ip,
          a,
          t,
          Vi
        ), (t.mode & ta) !== Bt && fc(t), a !== null) {
          at = a;
          return;
        }
        if (t = t.sibling, t !== null) {
          at = t;
          return;
        }
        at = t = e;
      } while (t !== null);
      ul === Kc && (ul = ub);
    }
    function vd(e, t) {
      do {
        var a = Pp(e.alternate, e);
        if (a !== null) {
          a.flags &= 32767, at = a;
          return;
        }
        if ((e.mode & ta) !== Bt) {
          fc(e), a = e.actualDuration;
          for (var i = e.child; i !== null; )
            a += i.actualDuration, i = i.sibling;
          e.actualDuration = a;
        }
        if (a = e.return, a !== null && (a.flags |= 32768, a.subtreeFlags = 0, a.deletions = null), !t && (e = e.sibling, e !== null)) {
          at = e;
          return;
        }
        at = e = a;
      } while (e !== null);
      ul = $v, at = null;
    }
    function gd(e, t, a, i, o, f, d, h, v) {
      e.cancelPendingCommit = null;
      do
        xo();
      while (aa !== Is);
      if (Ju.flushLegacyContextWarning(), Ju.flushPendingUnsafeLifecycleWarnings(), (Et & (qa | Wu)) !== An)
        throw Error("Should not already be working.");
      if (fe !== null && typeof fe.markCommitStarted == "function" && fe.markCommitStarted(a), t === null) He();
      else {
        if (a === 0 && console.error(
          "finishedLanes should not be empty during a commit. This is a bug in React."
        ), t === e.current)
          throw Error(
            "Cannot commit the same tree as before. This error is likely caused by a bug in React. Please file an issue."
          );
        if (f = t.lanes | t.childLanes, f |= kg, cr(
          e,
          a,
          f,
          d,
          h,
          v
        ), e === xt && (at = xt = null, ut = 0), Oh = t, hf = e, yf = a, E0 = f, A0 = o, db = i, (t.subtreeFlags & 10256) !== 0 || (t.flags & 10256) !== 0 ? (e.callbackNode = null, e.callbackPriority = 0, ky(Wo, function() {
          return ms(), null;
        })) : (e.callbackNode = null, e.callbackPriority = 0), Bv = sh(), i = (t.flags & 13878) !== 0, (t.subtreeFlags & 13878) !== 0 || i) {
          i = L.T, L.T = null, o = Ce.p, Ce.p = ql, d = Et, Et |= Wu;
          try {
            cd(e, t, a);
          } finally {
            Et = d, Ce.p = o, L.T = i;
          }
        }
        aa = fb, kn(), bd(), cv();
      }
    }
    function kn() {
      if (aa === fb) {
        aa = Is;
        var e = hf, t = Oh, a = yf, i = (t.flags & 13878) !== 0;
        if ((t.subtreeFlags & 13878) !== 0 || i) {
          i = L.T, L.T = null;
          var o = Ce.p;
          Ce.p = ql;
          var f = Et;
          Et |= Wu;
          try {
            Sh = a, Th = e, qy(t, e), Th = Sh = null, a = N0;
            var d = Bp(e.containerInfo), h = a.focusedElem, v = a.selectionRange;
            if (d !== h && h && h.ownerDocument && jp(
              h.ownerDocument.documentElement,
              h
            )) {
              if (v !== null && Zh(h)) {
                var b = v.start, B = v.end;
                if (B === void 0 && (B = b), "selectionStart" in h)
                  h.selectionStart = b, h.selectionEnd = Math.min(
                    B,
                    h.value.length
                  );
                else {
                  var X = h.ownerDocument || document, N = X && X.defaultView || window;
                  if (N.getSelection) {
                    var Q = N.getSelection(), me = h.textContent.length, xe = Math.min(
                      v.start,
                      me
                    ), Ht = v.end === void 0 ? xe : Math.min(v.end, me);
                    !Q.extend && xe > Ht && (d = Ht, Ht = xe, xe = d);
                    var ft = Qh(
                      h,
                      xe
                    ), T = Qh(
                      h,
                      Ht
                    );
                    if (ft && T && (Q.rangeCount !== 1 || Q.anchorNode !== ft.node || Q.anchorOffset !== ft.offset || Q.focusNode !== T.node || Q.focusOffset !== T.offset)) {
                      var E = X.createRange();
                      E.setStart(ft.node, ft.offset), Q.removeAllRanges(), xe > Ht ? (Q.addRange(E), Q.extend(T.node, T.offset)) : (E.setEnd(T.node, T.offset), Q.addRange(E));
                    }
                  }
                }
              }
              for (X = [], Q = h; Q = Q.parentNode; )
                Q.nodeType === 1 && X.push({
                  element: Q,
                  left: Q.scrollLeft,
                  top: Q.scrollTop
                });
              for (typeof h.focus == "function" && h.focus(), h = 0; h < X.length; h++) {
                var A = X[h];
                A.element.scrollLeft = A.left, A.element.scrollTop = A.top;
              }
            }
            hg = !!H0, N0 = H0 = null;
          } finally {
            Et = f, Ce.p = o, L.T = i;
          }
        }
        e.current = t, aa = sb;
      }
    }
    function bd() {
      if (aa === sb) {
        aa = Is;
        var e = hf, t = Oh, a = yf, i = (t.flags & 8772) !== 0;
        if ((t.subtreeFlags & 8772) !== 0 || i) {
          i = L.T, L.T = null;
          var o = Ce.p;
          Ce.p = ql;
          var f = Et;
          Et |= Wu;
          try {
            fe !== null && typeof fe.markLayoutEffectsStarted == "function" && fe.markLayoutEffectsStarted(a), Sh = a, Th = e, Ny(
              e,
              t.alternate,
              t
            ), Th = Sh = null, fe !== null && typeof fe.markLayoutEffectsStopped == "function" && fe.markLayoutEffectsStopped();
          } finally {
            Et = f, Ce.p = o, L.T = i;
          }
        }
        aa = rb;
      }
    }
    function cv() {
      if (aa === WS || aa === rb) {
        aa = Is, jg();
        var e = hf, t = Oh, a = yf, i = db, o = (t.subtreeFlags & 10256) !== 0 || (t.flags & 10256) !== 0;
        o ? aa = T0 : (aa = Is, Oh = hf = null, $n(e, e.pendingLanes), Ps = 0, hp = null);
        var f = e.pendingLanes;
        if (f === 0 && (df = null), o || No(e), o = to(a), t = t.stateNode, wl && typeof wl.onCommitFiberRoot == "function")
          try {
            var d = (t.current.flags & 128) === 128;
            switch (o) {
              case ql:
                var h = Vd;
                break;
              case En:
                h = Cs;
                break;
              case Xu:
                h = Wo;
                break;
              case Kd:
                h = xs;
                break;
              default:
                h = Wo;
            }
            wl.onCommitFiberRoot(
              Hi,
              t,
              h,
              d
            );
          } catch (X) {
            va || (va = !0, console.error(
              "React instrumentation encountered an error: %s",
              X
            ));
          }
        if (Ft && e.memoizedUpdaters.clear(), Yy(), i !== null) {
          d = L.T, h = Ce.p, Ce.p = ql, L.T = null;
          try {
            var v = e.onRecoverableError;
            for (t = 0; t < i.length; t++) {
              var b = i[t], B = ov(b.stack);
              ye(
                b.source,
                v,
                b.value,
                B
              );
            }
          } finally {
            L.T = d, Ce.p = h;
          }
        }
        (yf & 3) !== 0 && xo(), $a(e), f = e.pendingLanes, (a & 4194090) !== 0 && (f & 42) !== 0 ? (Gv = !0, e === R0 ? dp++ : (dp = 0, R0 = e)) : dp = 0, Dc(0), He();
      }
    }
    function ov(e) {
      return e = { componentStack: e }, Object.defineProperty(e, "digest", {
        get: function() {
          console.error(
            'You are accessing "digest" from the errorInfo object passed to onRecoverableError. This property is no longer provided as part of errorInfo but can be accessed as a property of the Error instance itself.'
          );
        }
      }), e;
    }
    function $n(e, t) {
      (e.pooledCacheLanes &= t) === 0 && (t = e.pooledCache, t != null && (e.pooledCache = null, Hn(t)));
    }
    function xo(e) {
      return kn(), bd(), cv(), ms();
    }
    function ms() {
      if (aa !== T0) return !1;
      var e = hf, t = E0;
      E0 = 0;
      var a = to(yf), i = Xu > a ? Xu : a;
      a = L.T;
      var o = Ce.p;
      try {
        Ce.p = i, L.T = null, i = A0, A0 = null;
        var f = hf, d = yf;
        if (aa = Is, Oh = hf = null, yf = 0, (Et & (qa | Wu)) !== An)
          throw Error("Cannot flush passive effects while already rendering.");
        O0 = !0, Iv = !1, fe !== null && typeof fe.markPassiveEffectsStarted == "function" && fe.markPassiveEffectsStarted(d);
        var h = Et;
        if (Et |= Wu, jy(f.current), fs(
          f,
          f.current,
          d,
          i
        ), fe !== null && typeof fe.markPassiveEffectsStopped == "function" && fe.markPassiveEffectsStopped(), No(f), Et = h, Dc(0, !1), Iv ? f === hp ? Ps++ : (Ps = 0, hp = f) : Ps = 0, Iv = O0 = !1, wl && typeof wl.onPostCommitFiberRoot == "function")
          try {
            wl.onPostCommitFiberRoot(Hi, f);
          } catch (b) {
            va || (va = !0, console.error(
              "React instrumentation encountered an error: %s",
              b
            ));
          }
        var v = f.current.stateNode;
        return v.effectDuration = 0, v.passiveEffectDuration = 0, !0;
      } finally {
        Ce.p = o, L.T = a, $n(e, t);
      }
    }
    function Ho(e, t, a) {
      t = Ra(a, t), t = Ll(e.stateNode, t, 2), e = hn(e, t, 2), e !== null && (gu(e, 2), $a(e));
    }
    function Me(e, t, a) {
      if (Dh = !1, e.tag === 3)
        Ho(e, e, a);
      else {
        for (; t !== null; ) {
          if (t.tag === 3) {
            Ho(
              t,
              e,
              a
            );
            return;
          }
          if (t.tag === 1) {
            var i = t.stateNode;
            if (typeof t.type.getDerivedStateFromError == "function" || typeof i.componentDidCatch == "function" && (df === null || !df.has(i))) {
              e = Ra(a, e), a = Zt(2), i = hn(t, a, 2), i !== null && (Pf(
                a,
                i,
                t,
                e
              ), gu(i, 2), $a(i));
              return;
            }
          }
          t = t.return;
        }
        console.error(
          `Internal React error: Attempted to capture a commit phase error inside a detached tree. This indicates a bug in React. Potential causes include deleting the same fiber more than once, committing an already-finished tree, or an inconsistent return pointer.

Error message:

%s`,
          a
        );
      }
    }
    function Zy(e, t, a) {
      var i = e.pingCache;
      if (i === null) {
        i = e.pingCache = new KS();
        var o = /* @__PURE__ */ new Set();
        i.set(t, o);
      } else
        o = i.get(t), o === void 0 && (o = /* @__PURE__ */ new Set(), i.set(t, o));
      o.has(a) || (g0 = !0, o.add(a), i = Ug.bind(null, e, t, a), Ft && wo(e, a), t.then(i, i));
    }
    function Ug(e, t, a) {
      var i = e.pingCache;
      i !== null && i.delete(t), e.pingedLanes |= e.suspendedLanes & a, e.warmLanes &= ~a, Gy() && L.actQueue === null && console.error(
        `A suspended resource finished loading inside a test, but the event was not wrapped in act(...).

When testing, code that resolves suspended data should be wrapped into act(...):

act(() => {
  /* finish loading suspended data */
});
/* assert on the output */

This ensures that you're testing the behavior the user would see in the browser. Learn more at https://react.dev/link/wrap-tests-with-act`
      ), xt === e && (ut & a) === a && (ul === ks || ul === m0 && (ut & 62914560) === ut && nu() - S0 < cb ? (Et & qa) === An && Oc(e, 0) : b0 |= a, Fs === ut && (Fs = 0)), $a(e);
    }
    function Ky(e, t) {
      t === 0 && (t = Mn()), e = ia(e, t), e !== null && (gu(e, t), $a(e));
    }
    function ps(e) {
      var t = e.memoizedState, a = 0;
      t !== null && (a = t.retryLane), Ky(e, a);
    }
    function Ri(e, t) {
      var a = 0;
      switch (e.tag) {
        case 13:
          var i = e.stateNode, o = e.memoizedState;
          o !== null && (a = o.retryLane);
          break;
        case 19:
          i = e.stateNode;
          break;
        case 22:
          i = e.stateNode._retryCache;
          break;
        default:
          throw Error(
            "Pinged unknown suspense boundary type. This is probably a bug in React."
          );
      }
      i !== null && i.delete(t), Ky(e, a);
    }
    function Sd(e, t, a) {
      if ((t.subtreeFlags & 67117056) !== 0)
        for (t = t.child; t !== null; ) {
          var i = e, o = t, f = o.type === Zo;
          f = a || f, o.tag !== 22 ? o.flags & 67108864 ? f && ye(
            o,
            Jy,
            i,
            o,
            (o.mode & a1) === Bt
          ) : Sd(
            i,
            o,
            f
          ) : o.memoizedState === null && (f && o.flags & 8192 ? ye(
            o,
            Jy,
            i,
            o
          ) : o.subtreeFlags & 67108864 && ye(
            o,
            Sd,
            i,
            o,
            f
          )), t = t.sibling;
        }
    }
    function Jy(e, t) {
      var a = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : !0;
      oe(!0);
      try {
        Ma(t), a && ds(t), Hu(e, t.alternate, t, !1), a && fd(e, t, 0, null, !1, 0);
      } finally {
        oe(!1);
      }
    }
    function No(e) {
      var t = !0;
      e.current.mode & (Sa | Ku) || (t = !1), Sd(
        e,
        e.current,
        t
      );
    }
    function Sn(e) {
      if ((Et & qa) === An) {
        var t = e.tag;
        if (t === 3 || t === 1 || t === 0 || t === 11 || t === 14 || t === 15) {
          if (t = re(e) || "ReactComponent", Pv !== null) {
            if (Pv.has(t)) return;
            Pv.add(t);
          } else Pv = /* @__PURE__ */ new Set([t]);
          ye(e, function() {
            console.error(
              "Can't perform a React state update on a component that hasn't mounted yet. This indicates that you have a side-effect in your render function that asynchronously later calls tries to update the component. Move this work to useEffect instead."
            );
          });
        }
      }
    }
    function wo(e, t) {
      Ft && e.memoizedUpdaters.forEach(function(a) {
        Ba(e, a, t);
      });
    }
    function ky(e, t) {
      var a = L.actQueue;
      return a !== null ? (a.push(t), PS) : Ld(e, t);
    }
    function fv(e) {
      Gy() && L.actQueue === null && ye(e, function() {
        console.error(
          `An update to %s inside a test was not wrapped in act(...).

When testing, code that causes React state updates should be wrapped into act(...):

act(() => {
  /* fire events that update state */
});
/* assert on the output */

This ensures that you're testing the behavior the user would see in the browser. Learn more at https://react.dev/link/wrap-tests-with-act`,
          re(e)
        );
      });
    }
    function $a(e) {
      e !== zh && e.next === null && (zh === null ? eg = zh = e : zh = zh.next = e), tg = !0, L.actQueue !== null ? z0 || (z0 = !0, dl()) : D0 || (D0 = !0, dl());
    }
    function Dc(e, t) {
      if (!M0 && tg) {
        M0 = !0;
        do
          for (var a = !1, i = eg; i !== null; ) {
            if (e !== 0) {
              var o = i.pendingLanes;
              if (o === 0) var f = 0;
              else {
                var d = i.suspendedLanes, h = i.pingedLanes;
                f = (1 << 31 - Ql(42 | e) + 1) - 1, f &= o & ~(d & ~h), f = f & 201326741 ? f & 201326741 | 1 : f ? f | 2 : 0;
              }
              f !== 0 && (a = !0, Ad(i, f));
            } else
              f = ut, f = pl(
                i,
                i === xt ? f : 0,
                i.cancelPendingCommit !== null || i.timeoutHandle !== lr
              ), (f & 3) === 0 || Iu(i, f) || (a = !0, Ad(i, f));
            i = i.next;
          }
        while (a);
        M0 = !1;
      }
    }
    function Td() {
      Ed();
    }
    function Ed() {
      tg = z0 = D0 = !1;
      var e = 0;
      er !== 0 && (qo() && (e = er), er = 0);
      for (var t = nu(), a = null, i = eg; i !== null; ) {
        var o = i.next, f = Wn(i, t);
        f === 0 ? (i.next = null, a === null ? eg = o : a.next = o, o === null && (zh = a)) : (a = i, (e !== 0 || (f & 3) !== 0) && (tg = !0)), i = o;
      }
      Dc(e);
    }
    function Wn(e, t) {
      for (var a = e.suspendedLanes, i = e.pingedLanes, o = e.expirationTimes, f = e.pendingLanes & -62914561; 0 < f; ) {
        var d = 31 - Ql(f), h = 1 << d, v = o[d];
        v === -1 ? ((h & a) === 0 || (h & i) !== 0) && (o[d] = ir(h, t)) : v <= t && (e.expiredLanes |= h), f &= ~h;
      }
      if (t = xt, a = ut, a = pl(
        e,
        e === t ? a : 0,
        e.cancelPendingCommit !== null || e.timeoutHandle !== lr
      ), i = e.callbackNode, a === 0 || e === t && (Mt === $s || Mt === Ws) || e.cancelPendingCommit !== null)
        return i !== null && Rd(i), e.callbackNode = null, e.callbackPriority = 0;
      if ((a & 3) === 0 || Iu(e, a)) {
        if (t = a & -a, t !== e.callbackPriority || L.actQueue !== null && i !== _0)
          Rd(i);
        else return t;
        switch (to(a)) {
          case ql:
          case En:
            a = Cs;
            break;
          case Xu:
            a = Wo;
            break;
          case Kd:
            a = xs;
            break;
          default:
            a = Wo;
        }
        return i = Jt.bind(null, e), L.actQueue !== null ? (L.actQueue.push(i), a = _0) : a = Ld(a, i), e.callbackPriority = t, e.callbackNode = a, t;
      }
      return i !== null && Rd(i), e.callbackPriority = 2, e.callbackNode = null, 2;
    }
    function Jt(e, t) {
      if (Gv = Yv = !1, aa !== Is && aa !== T0)
        return e.callbackNode = null, e.callbackPriority = 0, null;
      var a = e.callbackNode;
      if (xo() && e.callbackNode !== a)
        return null;
      var i = ut;
      return i = pl(
        e,
        e === xt ? i : 0,
        e.cancelPendingCommit !== null || e.timeoutHandle !== lr
      ), i === 0 ? null : (rl(
        e,
        i,
        t
      ), Wn(e, nu()), e.callbackNode != null && e.callbackNode === a ? Jt.bind(null, e) : null);
    }
    function Ad(e, t) {
      if (xo()) return null;
      Yv = Gv, Gv = !1, rl(e, t, !0);
    }
    function Rd(e) {
      e !== _0 && e !== null && qg(e);
    }
    function dl() {
      L.actQueue !== null && L.actQueue.push(function() {
        return Ed(), null;
      }), fT(function() {
        (Et & (qa | Wu)) !== An ? Ld(
          Vd,
          Td
        ) : Ed();
      });
    }
    function $y() {
      return er === 0 && (er = Ke()), er;
    }
    function Wy(e) {
      return e == null || typeof e == "symbol" || typeof e == "boolean" ? null : typeof e == "function" ? e : (J(e, "action"), oo("" + e));
    }
    function Fy(e, t) {
      var a = t.ownerDocument.createElement("input");
      return a.name = t.name, a.value = t.value, e.id && a.setAttribute("form", e.id), t.parentNode.insertBefore(a, t), e = new FormData(e), a.parentNode.removeChild(a), e;
    }
    function qt(e, t, a, i, o) {
      if (t === "submit" && a && a.stateNode === o) {
        var f = Wy(
          (o[ga] || null).action
        ), d = i.submitter;
        d && (t = (t = d[ga] || null) ? Wy(t.formAction) : d.getAttribute("formAction"), t !== null && (f = t, d = null));
        var h = new Re(
          "action",
          "action",
          null,
          i,
          o
        );
        e.push({
          event: h,
          listeners: [
            {
              instance: null,
              listener: function() {
                if (i.defaultPrevented) {
                  if (er !== 0) {
                    var v = d ? Fy(
                      o,
                      d
                    ) : new FormData(o), b = {
                      pending: !0,
                      data: v,
                      method: o.method,
                      action: f
                    };
                    Object.freeze(b), hc(
                      a,
                      b,
                      null,
                      v
                    );
                  }
                } else
                  typeof f == "function" && (h.preventDefault(), v = d ? Fy(
                    o,
                    d
                  ) : new FormData(o), b = {
                    pending: !0,
                    data: v,
                    method: o.method,
                    action: f
                  }, Object.freeze(b), hc(
                    a,
                    b,
                    f,
                    v
                  ));
              },
              currentTarget: o
            }
          ]
        });
      }
    }
    function Cl(e, t, a) {
      e.currentTarget = a;
      try {
        t(e);
      } catch (i) {
        s0(i);
      }
      e.currentTarget = null;
    }
    function Fn(e, t) {
      t = (t & 4) !== 0;
      for (var a = 0; a < e.length; a++) {
        var i = e[a];
        e: {
          var o = void 0, f = i.event;
          if (i = i.listeners, t)
            for (var d = i.length - 1; 0 <= d; d--) {
              var h = i[d], v = h.instance, b = h.currentTarget;
              if (h = h.listener, v !== o && f.isPropagationStopped())
                break e;
              v !== null ? ye(
                v,
                Cl,
                f,
                h,
                b
              ) : Cl(f, h, b), o = v;
            }
          else
            for (d = 0; d < i.length; d++) {
              if (h = i[d], v = h.instance, b = h.currentTarget, h = h.listener, v !== o && f.isPropagationStopped())
                break e;
              v !== null ? ye(
                v,
                Cl,
                f,
                h,
                b
              ) : Cl(f, h, b), o = v;
            }
        }
      }
    }
    function Pe(e, t) {
      U0.has(e) || console.error(
        'Did not expect a listenToNonDelegatedEvent() call for "%s". This is a bug in React. Please file an issue.',
        e
      );
      var a = t[Rm];
      a === void 0 && (a = t[Rm] = /* @__PURE__ */ new Set());
      var i = e + "__bubble";
      a.has(i) || (Dd(t, e, 2, !1), a.add(i));
    }
    function Od(e, t, a) {
      U0.has(e) && !t && console.error(
        'Did not expect a listenToNativeEvent() call for "%s" in the bubble phase. This is a bug in React. Please file an issue.',
        e
      );
      var i = 0;
      t && (i |= 4), Dd(
        a,
        e,
        i,
        t
      );
    }
    function Iy(e) {
      if (!e[lg]) {
        e[lg] = !0, Ov.forEach(function(a) {
          a !== "selectionchange" && (U0.has(a) || Od(a, !1, e), Od(a, !0, e));
        });
        var t = e.nodeType === 9 ? e : e.ownerDocument;
        t === null || t[lg] || (t[lg] = !0, Od("selectionchange", !1, t));
      }
    }
    function Dd(e, t, a, i) {
      switch (jd(t)) {
        case ql:
          var o = Hg;
          break;
        case En:
          o = qd;
          break;
        default:
          o = Mi;
      }
      a = o.bind(
        null,
        t,
        a,
        e
      ), o = void 0, !H || t !== "touchstart" && t !== "touchmove" && t !== "wheel" || (o = !0), i ? o !== void 0 ? e.addEventListener(t, a, {
        capture: !0,
        passive: o
      }) : e.addEventListener(t, a, !0) : o !== void 0 ? e.addEventListener(t, a, {
        passive: o
      }) : e.addEventListener(
        t,
        a,
        !1
      );
    }
    function Fl(e, t, a, i, o) {
      var f = i;
      if ((t & 1) === 0 && (t & 2) === 0 && i !== null)
        e: for (; ; ) {
          if (i === null) return;
          var d = i.tag;
          if (d === 3 || d === 4) {
            var h = i.stateNode.containerInfo;
            if (h === o) break;
            if (d === 4)
              for (d = i.return; d !== null; ) {
                var v = d.tag;
                if ((v === 3 || v === 4) && d.stateNode.containerInfo === o)
                  return;
                d = d.return;
              }
            for (; h !== null; ) {
              if (d = ua(h), d === null) return;
              if (v = d.tag, v === 5 || v === 6 || v === 26 || v === 27) {
                i = f = d;
                continue e;
              }
              h = h.parentNode;
            }
          }
          i = i.return;
        }
      vr(function() {
        var b = f, B = Pi(a), X = [];
        e: {
          var N = l1.get(e);
          if (N !== void 0) {
            var Q = Re, me = e;
            switch (e) {
              case "keypress":
                if (fo(a) === 0) break e;
              case "keydown":
              case "keyup":
                Q = pS;
                break;
              case "focusin":
                me = "focus", Q = st;
                break;
              case "focusout":
                me = "blur", Q = st;
                break;
              case "beforeblur":
              case "afterblur":
                Q = st;
                break;
              case "click":
                if (a.button === 2) break e;
              case "auxclick":
              case "dblclick":
              case "mousedown":
              case "mousemove":
              case "mouseup":
              case "mouseout":
              case "mouseover":
              case "contextmenu":
                Q = We;
                break;
              case "drag":
              case "dragend":
              case "dragenter":
              case "dragexit":
              case "dragleave":
              case "dragover":
              case "dragstart":
              case "drop":
                Q = _e;
                break;
              case "touchcancel":
              case "touchend":
              case "touchmove":
              case "touchstart":
                Q = bS;
                break;
              case I0:
              case P0:
              case e1:
                Q = Lg;
                break;
              case t1:
                Q = TS;
                break;
              case "scroll":
              case "scrollend":
                Q = M;
                break;
              case "wheel":
                Q = AS;
                break;
              case "copy":
              case "cut":
              case "paste":
                Q = fS;
                break;
              case "gotpointercapture":
              case "lostpointercapture":
              case "pointercancel":
              case "pointerdown":
              case "pointermove":
              case "pointerout":
              case "pointerover":
              case "pointerup":
                Q = Q0;
                break;
              case "toggle":
              case "beforetoggle":
                Q = OS;
            }
            var xe = (t & 4) !== 0, Ht = !xe && (e === "scroll" || e === "scrollend"), ft = xe ? N !== null ? N + "Capture" : null : N;
            xe = [];
            for (var T = b, E; T !== null; ) {
              var A = T;
              if (E = A.stateNode, A = A.tag, A !== 5 && A !== 26 && A !== 27 || E === null || ft === null || (A = Tu(T, ft), A != null && xe.push(
                Il(
                  T,
                  A,
                  E
                )
              )), Ht) break;
              T = T.return;
            }
            0 < xe.length && (N = new Q(
              N,
              me,
              null,
              a,
              B
            ), X.push({
              event: N,
              listeners: xe
            }));
          }
        }
        if ((t & 7) === 0) {
          e: {
            if (N = e === "mouseover" || e === "pointerover", Q = e === "mouseout" || e === "pointerout", N && a !== s && (me = a.relatedTarget || a.fromElement) && (ua(me) || me[wi]))
              break e;
            if ((Q || N) && (N = B.window === B ? B : (N = B.ownerDocument) ? N.defaultView || N.parentWindow : window, Q ? (me = a.relatedTarget || a.toElement, Q = b, me = me ? ua(me) : null, me !== null && (Ht = nt(me), xe = me.tag, me !== Ht || xe !== 5 && xe !== 27 && xe !== 6) && (me = null)) : (Q = null, me = b), Q !== me)) {
              if (xe = We, A = "onMouseLeave", ft = "onMouseEnter", T = "mouse", (e === "pointerout" || e === "pointerover") && (xe = Q0, A = "onPointerLeave", ft = "onPointerEnter", T = "pointer"), Ht = Q == null ? N : un(Q), E = me == null ? N : un(me), N = new xe(
                A,
                T + "leave",
                Q,
                a,
                B
              ), N.target = Ht, N.relatedTarget = E, A = null, ua(B) === b && (xe = new xe(
                ft,
                T + "enter",
                me,
                a,
                B
              ), xe.target = E, xe.relatedTarget = Ht, A = xe), Ht = A, Q && me)
                t: {
                  for (xe = Q, ft = me, T = 0, E = xe; E; E = Tl(E))
                    T++;
                  for (E = 0, A = ft; A; A = Tl(A))
                    E++;
                  for (; 0 < T - E; )
                    xe = Tl(xe), T--;
                  for (; 0 < E - T; )
                    ft = Tl(ft), E--;
                  for (; T--; ) {
                    if (xe === ft || ft !== null && xe === ft.alternate)
                      break t;
                    xe = Tl(xe), ft = Tl(ft);
                  }
                  xe = null;
                }
              else xe = null;
              Q !== null && Py(
                X,
                N,
                Q,
                xe,
                !1
              ), me !== null && Ht !== null && Py(
                X,
                Ht,
                me,
                xe,
                !0
              );
            }
          }
          e: {
            if (N = b ? un(b) : window, Q = N.nodeName && N.nodeName.toLowerCase(), Q === "select" || Q === "input" && N.type === "file")
              var Z = Vh;
            else if (Hp(N))
              if (W0)
                Z = Dg;
              else {
                Z = Xh;
                var ne = Rg;
              }
            else
              Q = N.nodeName, !Q || Q.toLowerCase() !== "input" || N.type !== "checkbox" && N.type !== "radio" ? b && Ii(b.elementType) && (Z = Vh) : Z = Og;
            if (Z && (Z = Z(e, b))) {
              Tr(
                X,
                Z,
                a,
                B
              );
              break e;
            }
            ne && ne(e, N, b), e === "focusout" && b && N.type === "number" && b.memoizedProps.value != null && rr(N, "number", N.value);
          }
          switch (ne = b ? un(b) : window, e) {
            case "focusin":
              (Hp(ne) || ne.contentEditable === "true") && (lh = ne, Xg = b, Qm = null);
              break;
            case "focusout":
              Qm = Xg = lh = null;
              break;
            case "mousedown":
              Qg = !0;
              break;
            case "contextmenu":
            case "mouseup":
            case "dragend":
              Qg = !1, Yp(
                X,
                a,
                B
              );
              break;
            case "selectionchange":
              if (_S) break;
            case "keydown":
            case "keyup":
              Yp(
                X,
                a,
                B
              );
          }
          var Ve;
          if (Vg)
            e: {
              switch (e) {
                case "compositionstart":
                  var pe = "onCompositionStart";
                  break e;
                case "compositionend":
                  pe = "onCompositionEnd";
                  break e;
                case "compositionupdate":
                  pe = "onCompositionUpdate";
                  break e;
              }
              pe = void 0;
            }
          else
            th ? Wl(e, a) && (pe = "onCompositionEnd") : e === "keydown" && a.keyCode === Z0 && (pe = "onCompositionStart");
          pe && (K0 && a.locale !== "ko" && (th || pe !== "onCompositionStart" ? pe === "onCompositionEnd" && th && (Ve = Eu()) : ($ = B, q = "value" in $ ? $.value : $.textContent, th = !0)), ne = vs(
            b,
            pe
          ), 0 < ne.length && (pe = new X0(
            pe,
            e,
            null,
            a,
            B
          ), X.push({
            event: pe,
            listeners: ne
          }), Ve ? pe.data = Ve : (Ve = ni(a), Ve !== null && (pe.data = Ve)))), (Ve = zS ? Sr(e, a) : Uf(e, a)) && (pe = vs(
            b,
            "onBeforeInput"
          ), 0 < pe.length && (ne = new rS(
            "onBeforeInput",
            "beforeinput",
            null,
            a,
            B
          ), X.push({
            event: ne,
            listeners: pe
          }), ne.data = Ve)), qt(
            X,
            e,
            b,
            a,
            B
          );
        }
        Fn(X, t);
      });
    }
    function Il(e, t, a) {
      return {
        instance: e,
        listener: t,
        currentTarget: a
      };
    }
    function vs(e, t) {
      for (var a = t + "Capture", i = []; e !== null; ) {
        var o = e, f = o.stateNode;
        if (o = o.tag, o !== 5 && o !== 26 && o !== 27 || f === null || (o = Tu(e, a), o != null && i.unshift(
          Il(e, o, f)
        ), o = Tu(e, t), o != null && i.push(
          Il(e, o, f)
        )), e.tag === 3) return i;
        e = e.return;
      }
      return [];
    }
    function Tl(e) {
      if (e === null) return null;
      do
        e = e.return;
      while (e && e.tag !== 5 && e.tag !== 27);
      return e || null;
    }
    function Py(e, t, a, i, o) {
      for (var f = t._reactName, d = []; a !== null && a !== i; ) {
        var h = a, v = h.alternate, b = h.stateNode;
        if (h = h.tag, v !== null && v === i) break;
        h !== 5 && h !== 26 && h !== 27 || b === null || (v = b, o ? (b = Tu(a, f), b != null && d.unshift(
          Il(a, b, v)
        )) : o || (b = Tu(a, f), b != null && d.push(
          Il(a, b, v)
        ))), a = a.return;
      }
      d.length !== 0 && e.push({ event: t, listeners: d });
    }
    function In(e, t) {
      co(e, t), e !== "input" && e !== "textarea" && e !== "select" || t == null || t.value !== null || Gm || (Gm = !0, e === "select" && t.multiple ? console.error(
        "`value` prop on `%s` should not be null. Consider using an empty array when `multiple` is set to `true` to clear the component or `undefined` for uncontrolled components.",
        e
      ) : console.error(
        "`value` prop on `%s` should not be null. Consider using an empty string to clear the component or `undefined` for uncontrolled components.",
        e
      ));
      var a = {
        registrationNameDependencies: en,
        possibleRegistrationNames: wc
      };
      Ii(e) || typeof t.is == "string" || Yh(e, t, a), t.contentEditable && !t.suppressContentEditableWarning && t.children != null && console.error(
        "A component is `contentEditable` and contains `children` managed by React. It is now your responsibility to guarantee that none of those nodes are unexpectedly modified or duplicated. This is probably not intentional."
      );
    }
    function jt(e, t, a, i) {
      t !== a && (a = xl(a), xl(t) !== a && (i[e] = t));
    }
    function Oi(e, t, a) {
      t.forEach(function(i) {
        a[tm(i)] = i === "style" ? Mc(e) : e.getAttribute(i);
      });
    }
    function Wa(e, t) {
      t === !1 ? console.error(
        "Expected `%s` listener to be a function, instead got `false`.\n\nIf you used to conditionally omit it with %s={condition && value}, pass %s={condition ? value : undefined} instead.",
        e,
        e,
        e
      ) : console.error(
        "Expected `%s` listener to be a function, instead got a value of `%s` type.",
        e,
        typeof t
      );
    }
    function zd(e, t) {
      return e = e.namespaceURI === Ys || e.namespaceURI === af ? e.ownerDocument.createElementNS(
        e.namespaceURI,
        e.tagName
      ) : e.ownerDocument.createElement(e.tagName), e.innerHTML = t, e.innerHTML;
    }
    function xl(e) {
      return g(e) && (console.error(
        "The provided HTML markup uses a value of unsupported type %s. This value must be coerced to a string before using it here.",
        be(e)
      ), j(e)), (typeof e == "string" ? e : "" + e).replace(eT, `
`).replace(tT, "");
    }
    function em(e, t) {
      return t = xl(t), xl(e) === t;
    }
    function wu() {
    }
    function ht(e, t, a, i, o, f) {
      switch (a) {
        case "children":
          typeof i == "string" ? (Mf(i, t, !1), t === "body" || t === "textarea" && i === "" || Fi(e, i)) : (typeof i == "number" || typeof i == "bigint") && (Mf("" + i, t, !1), t !== "body" && Fi(e, "" + i));
          break;
        case "className":
          Be(e, "class", i);
          break;
        case "tabIndex":
          Be(e, "tabindex", i);
          break;
        case "dir":
        case "role":
        case "viewBox":
        case "width":
        case "height":
          Be(e, a, i);
          break;
        case "style":
          _f(e, i, f);
          break;
        case "data":
          if (t !== "object") {
            Be(e, "data", i);
            break;
          }
        case "src":
        case "href":
          if (i === "" && (t !== "a" || a !== "href")) {
            console.error(
              a === "src" ? 'An empty string ("") was passed to the %s attribute. This may cause the browser to download the whole page again over the network. To fix this, either do not render the element at all or pass null to %s instead of an empty string.' : 'An empty string ("") was passed to the %s attribute. To fix this, either do not render the element at all or pass null to %s instead of an empty string.',
              a,
              a
            ), e.removeAttribute(a);
            break;
          }
          if (i == null || typeof i == "function" || typeof i == "symbol" || typeof i == "boolean") {
            e.removeAttribute(a);
            break;
          }
          J(i, a), i = oo("" + i), e.setAttribute(a, i);
          break;
        case "action":
        case "formAction":
          if (i != null && (t === "form" ? a === "formAction" ? console.error(
            "You can only pass the formAction prop to <input> or <button>. Use the action prop on <form>."
          ) : typeof i == "function" && (o.encType == null && o.method == null || ug || (ug = !0, console.error(
            "Cannot specify a encType or method for a form that specifies a function as the action. React provides those automatically. They will get overridden."
          )), o.target == null || ng || (ng = !0, console.error(
            "Cannot specify a target for a form that specifies a function as the action. The function will always be executed in the same window."
          ))) : t === "input" || t === "button" ? a === "action" ? console.error(
            "You can only pass the action prop to <form>. Use the formAction prop on <input> or <button>."
          ) : t !== "input" || o.type === "submit" || o.type === "image" || ag ? t !== "button" || o.type == null || o.type === "submit" || ag ? typeof i == "function" && (o.name == null || vb || (vb = !0, console.error(
            'Cannot specify a "name" prop for a button that specifies a function as a formAction. React needs it to encode which action should be invoked. It will get overridden.'
          )), o.formEncType == null && o.formMethod == null || ug || (ug = !0, console.error(
            "Cannot specify a formEncType or formMethod for a button that specifies a function as a formAction. React provides those automatically. They will get overridden."
          )), o.formTarget == null || ng || (ng = !0, console.error(
            "Cannot specify a formTarget for a button that specifies a function as a formAction. The function will always be executed in the same window."
          ))) : (ag = !0, console.error(
            'A button can only specify a formAction along with type="submit" or no type.'
          )) : (ag = !0, console.error(
            'An input can only specify a formAction along with type="submit" or type="image".'
          )) : console.error(
            a === "action" ? "You can only pass the action prop to <form>." : "You can only pass the formAction prop to <input> or <button>."
          )), typeof i == "function") {
            e.setAttribute(
              a,
              "javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')"
            );
            break;
          } else
            typeof f == "function" && (a === "formAction" ? (t !== "input" && ht(e, t, "name", o.name, o, null), ht(
              e,
              t,
              "formEncType",
              o.formEncType,
              o,
              null
            ), ht(
              e,
              t,
              "formMethod",
              o.formMethod,
              o,
              null
            ), ht(
              e,
              t,
              "formTarget",
              o.formTarget,
              o,
              null
            )) : (ht(
              e,
              t,
              "encType",
              o.encType,
              o,
              null
            ), ht(e, t, "method", o.method, o, null), ht(
              e,
              t,
              "target",
              o.target,
              o,
              null
            )));
          if (i == null || typeof i == "symbol" || typeof i == "boolean") {
            e.removeAttribute(a);
            break;
          }
          J(i, a), i = oo("" + i), e.setAttribute(a, i);
          break;
        case "onClick":
          i != null && (typeof i != "function" && Wa(a, i), e.onclick = wu);
          break;
        case "onScroll":
          i != null && (typeof i != "function" && Wa(a, i), Pe("scroll", e));
          break;
        case "onScrollEnd":
          i != null && (typeof i != "function" && Wa(a, i), Pe("scrollend", e));
          break;
        case "dangerouslySetInnerHTML":
          if (i != null) {
            if (typeof i != "object" || !("__html" in i))
              throw Error(
                "`props.dangerouslySetInnerHTML` must be in the form `{__html: ...}`. Please visit https://react.dev/link/dangerously-set-inner-html for more information."
              );
            if (a = i.__html, a != null) {
              if (o.children != null)
                throw Error(
                  "Can only set one of `children` or `props.dangerouslySetInnerHTML`."
                );
              e.innerHTML = a;
            }
          }
          break;
        case "multiple":
          e.multiple = i && typeof i != "function" && typeof i != "symbol";
          break;
        case "muted":
          e.muted = i && typeof i != "function" && typeof i != "symbol";
          break;
        case "suppressContentEditableWarning":
        case "suppressHydrationWarning":
        case "defaultValue":
        case "defaultChecked":
        case "innerHTML":
        case "ref":
          break;
        case "autoFocus":
          break;
        case "xlinkHref":
          if (i == null || typeof i == "function" || typeof i == "boolean" || typeof i == "symbol") {
            e.removeAttribute("xlink:href");
            break;
          }
          J(i, a), a = oo("" + i), e.setAttributeNS(tr, "xlink:href", a);
          break;
        case "contentEditable":
        case "spellCheck":
        case "draggable":
        case "value":
        case "autoReverse":
        case "externalResourcesRequired":
        case "focusable":
        case "preserveAlpha":
          i != null && typeof i != "function" && typeof i != "symbol" ? (J(i, a), e.setAttribute(a, "" + i)) : e.removeAttribute(a);
          break;
        case "inert":
          i !== "" || ig[a] || (ig[a] = !0, console.error(
            "Received an empty string for a boolean attribute `%s`. This will treat the attribute as if it were false. Either pass `false` to silence this warning, or pass `true` if you used an empty string in earlier versions of React to indicate this attribute is true.",
            a
          ));
        case "allowFullScreen":
        case "async":
        case "autoPlay":
        case "controls":
        case "default":
        case "defer":
        case "disabled":
        case "disablePictureInPicture":
        case "disableRemotePlayback":
        case "formNoValidate":
        case "hidden":
        case "loop":
        case "noModule":
        case "noValidate":
        case "open":
        case "playsInline":
        case "readOnly":
        case "required":
        case "reversed":
        case "scoped":
        case "seamless":
        case "itemScope":
          i && typeof i != "function" && typeof i != "symbol" ? e.setAttribute(a, "") : e.removeAttribute(a);
          break;
        case "capture":
        case "download":
          i === !0 ? e.setAttribute(a, "") : i !== !1 && i != null && typeof i != "function" && typeof i != "symbol" ? (J(i, a), e.setAttribute(a, i)) : e.removeAttribute(a);
          break;
        case "cols":
        case "rows":
        case "size":
        case "span":
          i != null && typeof i != "function" && typeof i != "symbol" && !isNaN(i) && 1 <= i ? (J(i, a), e.setAttribute(a, i)) : e.removeAttribute(a);
          break;
        case "rowSpan":
        case "start":
          i == null || typeof i == "function" || typeof i == "symbol" || isNaN(i) ? e.removeAttribute(a) : (J(i, a), e.setAttribute(a, i));
          break;
        case "popover":
          Pe("beforetoggle", e), Pe("toggle", e), ct(e, "popover", i);
          break;
        case "xlinkActuate":
          ll(
            e,
            tr,
            "xlink:actuate",
            i
          );
          break;
        case "xlinkArcrole":
          ll(
            e,
            tr,
            "xlink:arcrole",
            i
          );
          break;
        case "xlinkRole":
          ll(
            e,
            tr,
            "xlink:role",
            i
          );
          break;
        case "xlinkShow":
          ll(
            e,
            tr,
            "xlink:show",
            i
          );
          break;
        case "xlinkTitle":
          ll(
            e,
            tr,
            "xlink:title",
            i
          );
          break;
        case "xlinkType":
          ll(
            e,
            tr,
            "xlink:type",
            i
          );
          break;
        case "xmlBase":
          ll(
            e,
            C0,
            "xml:base",
            i
          );
          break;
        case "xmlLang":
          ll(
            e,
            C0,
            "xml:lang",
            i
          );
          break;
        case "xmlSpace":
          ll(
            e,
            C0,
            "xml:space",
            i
          );
          break;
        case "is":
          f != null && console.error(
            'Cannot update the "is" prop after it has been initialized.'
          ), ct(e, "is", i);
          break;
        case "innerText":
        case "textContent":
          break;
        case "popoverTarget":
          gb || i == null || typeof i != "object" || (gb = !0, console.error(
            "The `popoverTarget` prop expects the ID of an Element as a string. Received %s instead.",
            i
          ));
        default:
          !(2 < a.length) || a[0] !== "o" && a[0] !== "O" || a[1] !== "n" && a[1] !== "N" ? (a = pr(a), ct(e, a, i)) : en.hasOwnProperty(a) && i != null && typeof i != "function" && Wa(a, i);
      }
    }
    function zc(e, t, a, i, o, f) {
      switch (a) {
        case "style":
          _f(e, i, f);
          break;
        case "dangerouslySetInnerHTML":
          if (i != null) {
            if (typeof i != "object" || !("__html" in i))
              throw Error(
                "`props.dangerouslySetInnerHTML` must be in the form `{__html: ...}`. Please visit https://react.dev/link/dangerously-set-inner-html for more information."
              );
            if (a = i.__html, a != null) {
              if (o.children != null)
                throw Error(
                  "Can only set one of `children` or `props.dangerouslySetInnerHTML`."
                );
              e.innerHTML = a;
            }
          }
          break;
        case "children":
          typeof i == "string" ? Fi(e, i) : (typeof i == "number" || typeof i == "bigint") && Fi(e, "" + i);
          break;
        case "onScroll":
          i != null && (typeof i != "function" && Wa(a, i), Pe("scroll", e));
          break;
        case "onScrollEnd":
          i != null && (typeof i != "function" && Wa(a, i), Pe("scrollend", e));
          break;
        case "onClick":
          i != null && (typeof i != "function" && Wa(a, i), e.onclick = wu);
          break;
        case "suppressContentEditableWarning":
        case "suppressHydrationWarning":
        case "innerHTML":
        case "ref":
          break;
        case "innerText":
        case "textContent":
          break;
        default:
          if (en.hasOwnProperty(a))
            i != null && typeof i != "function" && Wa(a, i);
          else
            e: {
              if (a[0] === "o" && a[1] === "n" && (o = a.endsWith("Capture"), t = a.slice(2, o ? a.length - 7 : void 0), f = e[ga] || null, f = f != null ? f[a] : null, typeof f == "function" && e.removeEventListener(t, f, o), typeof i == "function")) {
                typeof f != "function" && f !== null && (a in e ? e[a] = null : e.hasAttribute(a) && e.removeAttribute(a)), e.addEventListener(t, i, o);
                break e;
              }
              a in e ? e[a] = i : i === !0 ? e.setAttribute(a, "") : ct(e, a, i);
            }
      }
    }
    function kt(e, t, a) {
      switch (In(t, a), t) {
        case "div":
        case "span":
        case "svg":
        case "path":
        case "a":
        case "g":
        case "p":
        case "li":
          break;
        case "img":
          Pe("error", e), Pe("load", e);
          var i = !1, o = !1, f;
          for (f in a)
            if (a.hasOwnProperty(f)) {
              var d = a[f];
              if (d != null)
                switch (f) {
                  case "src":
                    i = !0;
                    break;
                  case "srcSet":
                    o = !0;
                    break;
                  case "children":
                  case "dangerouslySetInnerHTML":
                    throw Error(
                      t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                    );
                  default:
                    ht(e, t, f, d, a, null);
                }
            }
          o && ht(e, t, "srcSet", a.srcSet, a, null), i && ht(e, t, "src", a.src, a, null);
          return;
        case "input":
          ve("input", a), Pe("invalid", e);
          var h = f = d = o = null, v = null, b = null;
          for (i in a)
            if (a.hasOwnProperty(i)) {
              var B = a[i];
              if (B != null)
                switch (i) {
                  case "name":
                    o = B;
                    break;
                  case "type":
                    d = B;
                    break;
                  case "checked":
                    v = B;
                    break;
                  case "defaultChecked":
                    b = B;
                    break;
                  case "value":
                    f = B;
                    break;
                  case "defaultValue":
                    h = B;
                    break;
                  case "children":
                  case "dangerouslySetInnerHTML":
                    if (B != null)
                      throw Error(
                        t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                      );
                    break;
                  default:
                    ht(e, t, i, B, a, null);
                }
            }
          ei(e, a), Mp(
            e,
            f,
            h,
            v,
            b,
            d,
            o,
            !1
          ), bu(e);
          return;
        case "select":
          ve("select", a), Pe("invalid", e), i = d = f = null;
          for (o in a)
            if (a.hasOwnProperty(o) && (h = a[o], h != null))
              switch (o) {
                case "value":
                  f = h;
                  break;
                case "defaultValue":
                  d = h;
                  break;
                case "multiple":
                  i = h;
                default:
                  ht(
                    e,
                    t,
                    o,
                    h,
                    a,
                    null
                  );
              }
          Of(e, a), t = f, a = d, e.multiple = !!i, t != null ? Su(e, !!i, t, !1) : a != null && Su(e, !!i, a, !0);
          return;
        case "textarea":
          ve("textarea", a), Pe("invalid", e), f = o = i = null;
          for (d in a)
            if (a.hasOwnProperty(d) && (h = a[d], h != null))
              switch (d) {
                case "value":
                  i = h;
                  break;
                case "defaultValue":
                  o = h;
                  break;
                case "children":
                  f = h;
                  break;
                case "dangerouslySetInnerHTML":
                  if (h != null)
                    throw Error(
                      "`dangerouslySetInnerHTML` does not make sense on <textarea>."
                    );
                  break;
                default:
                  ht(
                    e,
                    t,
                    d,
                    h,
                    a,
                    null
                  );
              }
          _n(e, a), Nh(e, i, o, f), bu(e);
          return;
        case "option":
          Hh(e, a);
          for (v in a)
            if (a.hasOwnProperty(v) && (i = a[v], i != null))
              switch (v) {
                case "selected":
                  e.selected = i && typeof i != "function" && typeof i != "symbol";
                  break;
                default:
                  ht(e, t, v, i, a, null);
              }
          return;
        case "dialog":
          Pe("beforetoggle", e), Pe("toggle", e), Pe("cancel", e), Pe("close", e);
          break;
        case "iframe":
        case "object":
          Pe("load", e);
          break;
        case "video":
        case "audio":
          for (i = 0; i < yp.length; i++)
            Pe(yp[i], e);
          break;
        case "image":
          Pe("error", e), Pe("load", e);
          break;
        case "details":
          Pe("toggle", e);
          break;
        case "embed":
        case "source":
        case "link":
          Pe("error", e), Pe("load", e);
        case "area":
        case "base":
        case "br":
        case "col":
        case "hr":
        case "keygen":
        case "meta":
        case "param":
        case "track":
        case "wbr":
        case "menuitem":
          for (b in a)
            if (a.hasOwnProperty(b) && (i = a[b], i != null))
              switch (b) {
                case "children":
                case "dangerouslySetInnerHTML":
                  throw Error(
                    t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                  );
                default:
                  ht(e, t, b, i, a, null);
              }
          return;
        default:
          if (Ii(t)) {
            for (B in a)
              a.hasOwnProperty(B) && (i = a[B], i !== void 0 && zc(
                e,
                t,
                B,
                i,
                a,
                void 0
              ));
            return;
          }
      }
      for (h in a)
        a.hasOwnProperty(h) && (i = a[h], i != null && ht(e, t, h, i, a, null));
    }
    function sv(e, t, a, i) {
      switch (In(t, i), t) {
        case "div":
        case "span":
        case "svg":
        case "path":
        case "a":
        case "g":
        case "p":
        case "li":
          break;
        case "input":
          var o = null, f = null, d = null, h = null, v = null, b = null, B = null;
          for (Q in a) {
            var X = a[Q];
            if (a.hasOwnProperty(Q) && X != null)
              switch (Q) {
                case "checked":
                  break;
                case "value":
                  break;
                case "defaultValue":
                  v = X;
                default:
                  i.hasOwnProperty(Q) || ht(
                    e,
                    t,
                    Q,
                    null,
                    i,
                    X
                  );
              }
          }
          for (var N in i) {
            var Q = i[N];
            if (X = a[N], i.hasOwnProperty(N) && (Q != null || X != null))
              switch (N) {
                case "type":
                  f = Q;
                  break;
                case "name":
                  o = Q;
                  break;
                case "checked":
                  b = Q;
                  break;
                case "defaultChecked":
                  B = Q;
                  break;
                case "value":
                  d = Q;
                  break;
                case "defaultValue":
                  h = Q;
                  break;
                case "children":
                case "dangerouslySetInnerHTML":
                  if (Q != null)
                    throw Error(
                      t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                    );
                  break;
                default:
                  Q !== X && ht(
                    e,
                    t,
                    N,
                    Q,
                    i,
                    X
                  );
              }
          }
          t = a.type === "checkbox" || a.type === "radio" ? a.checked != null : a.value != null, i = i.type === "checkbox" || i.type === "radio" ? i.checked != null : i.value != null, t || !i || pb || (console.error(
            "A component is changing an uncontrolled input to be controlled. This is likely caused by the value changing from undefined to a defined value, which should not happen. Decide between using a controlled or uncontrolled input element for the lifetime of the component. More info: https://react.dev/link/controlled-components"
          ), pb = !0), !t || i || mb || (console.error(
            "A component is changing a controlled input to be uncontrolled. This is likely caused by the value changing from a defined to undefined, which should not happen. Decide between using a controlled or uncontrolled input element for the lifetime of the component. More info: https://react.dev/link/controlled-components"
          ), mb = !0), ti(
            e,
            d,
            h,
            v,
            b,
            B,
            f,
            o
          );
          return;
        case "select":
          Q = d = h = N = null;
          for (f in a)
            if (v = a[f], a.hasOwnProperty(f) && v != null)
              switch (f) {
                case "value":
                  break;
                case "multiple":
                  Q = v;
                default:
                  i.hasOwnProperty(f) || ht(
                    e,
                    t,
                    f,
                    null,
                    i,
                    v
                  );
              }
          for (o in i)
            if (f = i[o], v = a[o], i.hasOwnProperty(o) && (f != null || v != null))
              switch (o) {
                case "value":
                  N = f;
                  break;
                case "defaultValue":
                  h = f;
                  break;
                case "multiple":
                  d = f;
                default:
                  f !== v && ht(
                    e,
                    t,
                    o,
                    f,
                    i,
                    v
                  );
              }
          i = h, t = d, a = Q, N != null ? Su(e, !!t, N, !1) : !!a != !!t && (i != null ? Su(e, !!t, i, !0) : Su(e, !!t, t ? [] : "", !1));
          return;
        case "textarea":
          Q = N = null;
          for (h in a)
            if (o = a[h], a.hasOwnProperty(h) && o != null && !i.hasOwnProperty(h))
              switch (h) {
                case "value":
                  break;
                case "children":
                  break;
                default:
                  ht(e, t, h, null, i, o);
              }
          for (d in i)
            if (o = i[d], f = a[d], i.hasOwnProperty(d) && (o != null || f != null))
              switch (d) {
                case "value":
                  N = o;
                  break;
                case "defaultValue":
                  Q = o;
                  break;
                case "children":
                  break;
                case "dangerouslySetInnerHTML":
                  if (o != null)
                    throw Error(
                      "`dangerouslySetInnerHTML` does not make sense on <textarea>."
                    );
                  break;
                default:
                  o !== f && ht(e, t, d, o, i, f);
              }
          dr(e, N, Q);
          return;
        case "option":
          for (var me in a)
            if (N = a[me], a.hasOwnProperty(me) && N != null && !i.hasOwnProperty(me))
              switch (me) {
                case "selected":
                  e.selected = !1;
                  break;
                default:
                  ht(
                    e,
                    t,
                    me,
                    null,
                    i,
                    N
                  );
              }
          for (v in i)
            if (N = i[v], Q = a[v], i.hasOwnProperty(v) && N !== Q && (N != null || Q != null))
              switch (v) {
                case "selected":
                  e.selected = N && typeof N != "function" && typeof N != "symbol";
                  break;
                default:
                  ht(
                    e,
                    t,
                    v,
                    N,
                    i,
                    Q
                  );
              }
          return;
        case "img":
        case "link":
        case "area":
        case "base":
        case "br":
        case "col":
        case "embed":
        case "hr":
        case "keygen":
        case "meta":
        case "param":
        case "source":
        case "track":
        case "wbr":
        case "menuitem":
          for (var xe in a)
            N = a[xe], a.hasOwnProperty(xe) && N != null && !i.hasOwnProperty(xe) && ht(
              e,
              t,
              xe,
              null,
              i,
              N
            );
          for (b in i)
            if (N = i[b], Q = a[b], i.hasOwnProperty(b) && N !== Q && (N != null || Q != null))
              switch (b) {
                case "children":
                case "dangerouslySetInnerHTML":
                  if (N != null)
                    throw Error(
                      t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                    );
                  break;
                default:
                  ht(
                    e,
                    t,
                    b,
                    N,
                    i,
                    Q
                  );
              }
          return;
        default:
          if (Ii(t)) {
            for (var Ht in a)
              N = a[Ht], a.hasOwnProperty(Ht) && N !== void 0 && !i.hasOwnProperty(Ht) && zc(
                e,
                t,
                Ht,
                void 0,
                i,
                N
              );
            for (B in i)
              N = i[B], Q = a[B], !i.hasOwnProperty(B) || N === Q || N === void 0 && Q === void 0 || zc(
                e,
                t,
                B,
                N,
                i,
                Q
              );
            return;
          }
      }
      for (var ft in a)
        N = a[ft], a.hasOwnProperty(ft) && N != null && !i.hasOwnProperty(ft) && ht(e, t, ft, null, i, N);
      for (X in i)
        N = i[X], Q = a[X], !i.hasOwnProperty(X) || N === Q || N == null && Q == null || ht(e, t, X, N, i, Q);
    }
    function tm(e) {
      switch (e) {
        case "class":
          return "className";
        case "for":
          return "htmlFor";
        default:
          return e;
      }
    }
    function Mc(e) {
      var t = {};
      e = e.style;
      for (var a = 0; a < e.length; a++) {
        var i = e[a];
        t[i] = e.getPropertyValue(i);
      }
      return t;
    }
    function lm(e, t, a) {
      if (t != null && typeof t != "object")
        console.error(
          "The `style` prop expects a mapping from style properties to values, not a string. For example, style={{marginRight: spacing + 'em'}} when using JSX."
        );
      else {
        var i, o = i = "", f;
        for (f in t)
          if (t.hasOwnProperty(f)) {
            var d = t[f];
            d != null && typeof d != "boolean" && d !== "" && (f.indexOf("--") === 0 ? (I(d, f), i += o + f + ":" + ("" + d).trim()) : typeof d != "number" || d === 0 || Bs.has(f) ? (I(d, f), i += o + f.replace(Qu, "-$1").toLowerCase().replace(Zu, "-ms-") + ":" + ("" + d).trim()) : i += o + f.replace(Qu, "-$1").toLowerCase().replace(Zu, "-ms-") + ":" + d + "px", o = ";");
          }
        i = i || null, t = e.getAttribute("style"), t !== i && (i = xl(i), xl(t) !== i && (a.style = Mc(e)));
      }
    }
    function Pl(e, t, a, i, o, f) {
      if (o.delete(a), e = e.getAttribute(a), e === null)
        switch (typeof i) {
          case "undefined":
          case "function":
          case "symbol":
          case "boolean":
            return;
        }
      else if (i != null)
        switch (typeof i) {
          case "function":
          case "symbol":
          case "boolean":
            break;
          default:
            if (J(i, t), e === "" + i)
              return;
        }
      jt(t, e, i, f);
    }
    function am(e, t, a, i, o, f) {
      if (o.delete(a), e = e.getAttribute(a), e === null) {
        switch (typeof i) {
          case "function":
          case "symbol":
            return;
        }
        if (!i) return;
      } else
        switch (typeof i) {
          case "function":
          case "symbol":
            break;
          default:
            if (i) return;
        }
      jt(t, e, i, f);
    }
    function nm(e, t, a, i, o, f) {
      if (o.delete(a), e = e.getAttribute(a), e === null)
        switch (typeof i) {
          case "undefined":
          case "function":
          case "symbol":
            return;
        }
      else if (i != null)
        switch (typeof i) {
          case "function":
          case "symbol":
            break;
          default:
            if (J(i, a), e === "" + i)
              return;
        }
      jt(t, e, i, f);
    }
    function rv(e, t, a, i, o, f) {
      if (o.delete(a), e = e.getAttribute(a), e === null)
        switch (typeof i) {
          case "undefined":
          case "function":
          case "symbol":
          case "boolean":
            return;
          default:
            if (isNaN(i)) return;
        }
      else if (i != null)
        switch (typeof i) {
          case "function":
          case "symbol":
          case "boolean":
            break;
          default:
            if (!isNaN(i) && (J(i, t), e === "" + i))
              return;
        }
      jt(t, e, i, f);
    }
    function gt(e, t, a, i, o, f) {
      if (o.delete(a), e = e.getAttribute(a), e === null)
        switch (typeof i) {
          case "undefined":
          case "function":
          case "symbol":
          case "boolean":
            return;
        }
      else if (i != null)
        switch (typeof i) {
          case "function":
          case "symbol":
          case "boolean":
            break;
          default:
            if (J(i, t), a = oo("" + i), e === a)
              return;
        }
      jt(t, e, i, f);
    }
    function _t(e, t, a, i) {
      for (var o = {}, f = /* @__PURE__ */ new Set(), d = e.attributes, h = 0; h < d.length; h++)
        switch (d[h].name.toLowerCase()) {
          case "value":
            break;
          case "checked":
            break;
          case "selected":
            break;
          default:
            f.add(d[h].name);
        }
      if (Ii(t)) {
        for (var v in a)
          if (a.hasOwnProperty(v)) {
            var b = a[v];
            if (b != null) {
              if (en.hasOwnProperty(v))
                typeof b != "function" && Wa(v, b);
              else if (a.suppressHydrationWarning !== !0)
                switch (v) {
                  case "children":
                    typeof b != "string" && typeof b != "number" || jt(
                      "children",
                      e.textContent,
                      b,
                      o
                    );
                    continue;
                  case "suppressContentEditableWarning":
                  case "suppressHydrationWarning":
                  case "defaultValue":
                  case "defaultChecked":
                  case "innerHTML":
                  case "ref":
                    continue;
                  case "dangerouslySetInnerHTML":
                    d = e.innerHTML, b = b ? b.__html : void 0, b != null && (b = zd(e, b), jt(
                      v,
                      d,
                      b,
                      o
                    ));
                    continue;
                  case "style":
                    f.delete(v), lm(e, b, o);
                    continue;
                  case "offsetParent":
                  case "offsetTop":
                  case "offsetLeft":
                  case "offsetWidth":
                  case "offsetHeight":
                  case "isContentEditable":
                  case "outerText":
                  case "outerHTML":
                    f.delete(v.toLowerCase()), console.error(
                      "Assignment to read-only property will result in a no-op: `%s`",
                      v
                    );
                    continue;
                  case "className":
                    f.delete("class"), d = Ye(
                      e,
                      "class",
                      b
                    ), jt(
                      "className",
                      d,
                      b,
                      o
                    );
                    continue;
                  default:
                    i.context === kc && t !== "svg" && t !== "math" ? f.delete(v.toLowerCase()) : f.delete(v), d = Ye(
                      e,
                      v,
                      b
                    ), jt(
                      v,
                      d,
                      b,
                      o
                    );
                }
            }
          }
      } else
        for (b in a)
          if (a.hasOwnProperty(b) && (v = a[b], v != null)) {
            if (en.hasOwnProperty(b))
              typeof v != "function" && Wa(b, v);
            else if (a.suppressHydrationWarning !== !0)
              switch (b) {
                case "children":
                  typeof v != "string" && typeof v != "number" || jt(
                    "children",
                    e.textContent,
                    v,
                    o
                  );
                  continue;
                case "suppressContentEditableWarning":
                case "suppressHydrationWarning":
                case "value":
                case "checked":
                case "selected":
                case "defaultValue":
                case "defaultChecked":
                case "innerHTML":
                case "ref":
                  continue;
                case "dangerouslySetInnerHTML":
                  d = e.innerHTML, v = v ? v.__html : void 0, v != null && (v = zd(e, v), d !== v && (o[b] = { __html: d }));
                  continue;
                case "className":
                  Pl(
                    e,
                    b,
                    "class",
                    v,
                    f,
                    o
                  );
                  continue;
                case "tabIndex":
                  Pl(
                    e,
                    b,
                    "tabindex",
                    v,
                    f,
                    o
                  );
                  continue;
                case "style":
                  f.delete(b), lm(e, v, o);
                  continue;
                case "multiple":
                  f.delete(b), jt(
                    b,
                    e.multiple,
                    v,
                    o
                  );
                  continue;
                case "muted":
                  f.delete(b), jt(
                    b,
                    e.muted,
                    v,
                    o
                  );
                  continue;
                case "autoFocus":
                  f.delete("autofocus"), jt(
                    b,
                    e.autofocus,
                    v,
                    o
                  );
                  continue;
                case "data":
                  if (t !== "object") {
                    f.delete(b), d = e.getAttribute("data"), jt(
                      b,
                      d,
                      v,
                      o
                    );
                    continue;
                  }
                case "src":
                case "href":
                  if (!(v !== "" || t === "a" && b === "href" || t === "object" && b === "data")) {
                    console.error(
                      b === "src" ? 'An empty string ("") was passed to the %s attribute. This may cause the browser to download the whole page again over the network. To fix this, either do not render the element at all or pass null to %s instead of an empty string.' : 'An empty string ("") was passed to the %s attribute. To fix this, either do not render the element at all or pass null to %s instead of an empty string.',
                      b,
                      b
                    );
                    continue;
                  }
                  gt(
                    e,
                    b,
                    b,
                    v,
                    f,
                    o
                  );
                  continue;
                case "action":
                case "formAction":
                  if (d = e.getAttribute(b), typeof v == "function") {
                    f.delete(b.toLowerCase()), b === "formAction" ? (f.delete("name"), f.delete("formenctype"), f.delete("formmethod"), f.delete("formtarget")) : (f.delete("enctype"), f.delete("method"), f.delete("target"));
                    continue;
                  } else if (d === lT) {
                    f.delete(b.toLowerCase()), jt(
                      b,
                      "function",
                      v,
                      o
                    );
                    continue;
                  }
                  gt(
                    e,
                    b,
                    b.toLowerCase(),
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkHref":
                  gt(
                    e,
                    b,
                    "xlink:href",
                    v,
                    f,
                    o
                  );
                  continue;
                case "contentEditable":
                  nm(
                    e,
                    b,
                    "contenteditable",
                    v,
                    f,
                    o
                  );
                  continue;
                case "spellCheck":
                  nm(
                    e,
                    b,
                    "spellcheck",
                    v,
                    f,
                    o
                  );
                  continue;
                case "draggable":
                case "autoReverse":
                case "externalResourcesRequired":
                case "focusable":
                case "preserveAlpha":
                  nm(
                    e,
                    b,
                    b,
                    v,
                    f,
                    o
                  );
                  continue;
                case "allowFullScreen":
                case "async":
                case "autoPlay":
                case "controls":
                case "default":
                case "defer":
                case "disabled":
                case "disablePictureInPicture":
                case "disableRemotePlayback":
                case "formNoValidate":
                case "hidden":
                case "loop":
                case "noModule":
                case "noValidate":
                case "open":
                case "playsInline":
                case "readOnly":
                case "required":
                case "reversed":
                case "scoped":
                case "seamless":
                case "itemScope":
                  am(
                    e,
                    b,
                    b.toLowerCase(),
                    v,
                    f,
                    o
                  );
                  continue;
                case "capture":
                case "download":
                  e: {
                    h = e;
                    var B = d = b, X = o;
                    if (f.delete(B), h = h.getAttribute(B), h === null)
                      switch (typeof v) {
                        case "undefined":
                        case "function":
                        case "symbol":
                          break e;
                        default:
                          if (v === !1) break e;
                      }
                    else if (v != null)
                      switch (typeof v) {
                        case "function":
                        case "symbol":
                          break;
                        case "boolean":
                          if (v === !0 && h === "") break e;
                          break;
                        default:
                          if (J(v, d), h === "" + v)
                            break e;
                      }
                    jt(
                      d,
                      h,
                      v,
                      X
                    );
                  }
                  continue;
                case "cols":
                case "rows":
                case "size":
                case "span":
                  e: {
                    if (h = e, B = d = b, X = o, f.delete(B), h = h.getAttribute(B), h === null)
                      switch (typeof v) {
                        case "undefined":
                        case "function":
                        case "symbol":
                        case "boolean":
                          break e;
                        default:
                          if (isNaN(v) || 1 > v) break e;
                      }
                    else if (v != null)
                      switch (typeof v) {
                        case "function":
                        case "symbol":
                        case "boolean":
                          break;
                        default:
                          if (!(isNaN(v) || 1 > v) && (J(v, d), h === "" + v))
                            break e;
                      }
                    jt(
                      d,
                      h,
                      v,
                      X
                    );
                  }
                  continue;
                case "rowSpan":
                  rv(
                    e,
                    b,
                    "rowspan",
                    v,
                    f,
                    o
                  );
                  continue;
                case "start":
                  rv(
                    e,
                    b,
                    b,
                    v,
                    f,
                    o
                  );
                  continue;
                case "xHeight":
                  Pl(
                    e,
                    b,
                    "x-height",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkActuate":
                  Pl(
                    e,
                    b,
                    "xlink:actuate",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkArcrole":
                  Pl(
                    e,
                    b,
                    "xlink:arcrole",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkRole":
                  Pl(
                    e,
                    b,
                    "xlink:role",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkShow":
                  Pl(
                    e,
                    b,
                    "xlink:show",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkTitle":
                  Pl(
                    e,
                    b,
                    "xlink:title",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkType":
                  Pl(
                    e,
                    b,
                    "xlink:type",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xmlBase":
                  Pl(
                    e,
                    b,
                    "xml:base",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xmlLang":
                  Pl(
                    e,
                    b,
                    "xml:lang",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xmlSpace":
                  Pl(
                    e,
                    b,
                    "xml:space",
                    v,
                    f,
                    o
                  );
                  continue;
                case "inert":
                  v !== "" || ig[b] || (ig[b] = !0, console.error(
                    "Received an empty string for a boolean attribute `%s`. This will treat the attribute as if it were false. Either pass `false` to silence this warning, or pass `true` if you used an empty string in earlier versions of React to indicate this attribute is true.",
                    b
                  )), am(
                    e,
                    b,
                    b,
                    v,
                    f,
                    o
                  );
                  continue;
                default:
                  if (!(2 < b.length) || b[0] !== "o" && b[0] !== "O" || b[1] !== "n" && b[1] !== "N") {
                    h = pr(b), d = !1, i.context === kc && t !== "svg" && t !== "math" ? f.delete(h.toLowerCase()) : (B = b.toLowerCase(), B = Bc.hasOwnProperty(
                      B
                    ) && Bc[B] || null, B !== null && B !== b && (d = !0, f.delete(B)), f.delete(h));
                    e: if (B = e, X = h, h = v, Ne(X))
                      if (B.hasAttribute(X))
                        B = B.getAttribute(
                          X
                        ), J(
                          h,
                          X
                        ), h = B === "" + h ? h : B;
                      else {
                        switch (typeof h) {
                          case "function":
                          case "symbol":
                            break e;
                          case "boolean":
                            if (B = X.toLowerCase().slice(0, 5), B !== "data-" && B !== "aria-")
                              break e;
                        }
                        h = h === void 0 ? void 0 : null;
                      }
                    else h = void 0;
                    d || jt(
                      b,
                      h,
                      v,
                      o
                    );
                  }
              }
          }
      return 0 < f.size && a.suppressHydrationWarning !== !0 && Oi(e, f, o), Object.keys(o).length === 0 ? null : o;
    }
    function tt(e, t) {
      switch (e.length) {
        case 0:
          return "";
        case 1:
          return e[0];
        case 2:
          return e[0] + " " + t + " " + e[1];
        default:
          return e.slice(0, -1).join(", ") + ", " + t + " " + e[e.length - 1];
      }
    }
    function lt(e) {
      return e.nodeType === 9 ? e : e.ownerDocument;
    }
    function St(e) {
      switch (e) {
        case af:
          return Mh;
        case Ys:
          return fg;
        default:
          return kc;
      }
    }
    function ya(e, t) {
      if (e === kc)
        switch (t) {
          case "svg":
            return Mh;
          case "math":
            return fg;
          default:
            return kc;
        }
      return e === Mh && t === "foreignObject" ? kc : e;
    }
    function Pn(e, t) {
      return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.children == "bigint" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null;
    }
    function qo() {
      var e = window.event;
      return e && e.type === "popstate" ? e === w0 ? !1 : (w0 = e, !0) : (w0 = null, !1);
    }
    function um(e) {
      setTimeout(function() {
        throw e;
      });
    }
    function qu(e, t, a) {
      switch (t) {
        case "button":
        case "input":
        case "select":
        case "textarea":
          a.autoFocus && e.focus();
          break;
        case "img":
          a.src ? e.src = a.src : a.srcSet && (e.srcset = a.srcSet);
      }
    }
    function $t(e, t, a, i) {
      sv(e, t, a, i), e[ga] = i;
    }
    function ju(e) {
      Fi(e, "");
    }
    function _c(e, t, a) {
      e.nodeValue = a;
    }
    function eu(e) {
      return e === "head";
    }
    function Fa(e, t) {
      e.removeChild(t);
    }
    function jo(e, t) {
      (e.nodeType === 9 ? e.body : e.nodeName === "HTML" ? e.ownerDocument.body : e).removeChild(t);
    }
    function Bo(e, t) {
      var a = t, i = 0, o = 0;
      do {
        var f = a.nextSibling;
        if (e.removeChild(a), f && f.nodeType === 8)
          if (a = f.data, a === og) {
            if (0 < i && 8 > i) {
              a = i;
              var d = e.ownerDocument;
              if (a & nT && Vo(d.documentElement), a & uT && Vo(d.body), a & iT)
                for (a = d.head, Vo(a), d = a.firstChild; d; ) {
                  var h = d.nextSibling, v = d.nodeName;
                  d[Po] || v === "SCRIPT" || v === "STYLE" || v === "LINK" && d.rel.toLowerCase() === "stylesheet" || a.removeChild(d), d = h;
                }
            }
            if (o === 0) {
              e.removeChild(f), Hc(t);
              return;
            }
            o--;
          } else
            a === cg || a === Jc || a === mp ? o++ : i = a.charCodeAt(0) - 48;
        else i = 0;
        a = f;
      } while (a);
      Hc(t);
    }
    function ma(e) {
      e = e.style, typeof e.setProperty == "function" ? e.setProperty("display", "none", "important") : e.display = "none";
    }
    function im(e) {
      e.nodeValue = "";
    }
    function cm(e, t) {
      t = t[cT], t = t != null && t.hasOwnProperty("display") ? t.display : null, e.style.display = t == null || typeof t == "boolean" ? "" : ("" + t).trim();
    }
    function Md(e, t) {
      e.nodeValue = t;
    }
    function Yo(e) {
      var t = e.firstChild;
      for (t && t.nodeType === 10 && (t = t.nextSibling); t; ) {
        var a = t;
        switch (t = t.nextSibling, a.nodeName) {
          case "HTML":
          case "HEAD":
          case "BODY":
            Yo(a), nn(a);
            continue;
          case "SCRIPT":
          case "STYLE":
            continue;
          case "LINK":
            if (a.rel.toLowerCase() === "stylesheet") continue;
        }
        e.removeChild(a);
      }
    }
    function Di(e, t, a, i) {
      for (; e.nodeType === 1; ) {
        var o = a;
        if (e.nodeName.toLowerCase() !== t.toLowerCase()) {
          if (!i && (e.nodeName !== "INPUT" || e.type !== "hidden"))
            break;
        } else if (i) {
          if (!e[Po])
            switch (t) {
              case "meta":
                if (!e.hasAttribute("itemprop")) break;
                return e;
              case "link":
                if (f = e.getAttribute("rel"), f === "stylesheet" && e.hasAttribute("data-precedence"))
                  break;
                if (f !== o.rel || e.getAttribute("href") !== (o.href == null || o.href === "" ? null : o.href) || e.getAttribute("crossorigin") !== (o.crossOrigin == null ? null : o.crossOrigin) || e.getAttribute("title") !== (o.title == null ? null : o.title))
                  break;
                return e;
              case "style":
                if (e.hasAttribute("data-precedence")) break;
                return e;
              case "script":
                if (f = e.getAttribute("src"), (f !== (o.src == null ? null : o.src) || e.getAttribute("type") !== (o.type == null ? null : o.type) || e.getAttribute("crossorigin") !== (o.crossOrigin == null ? null : o.crossOrigin)) && f && e.hasAttribute("async") && !e.hasAttribute("itemprop"))
                  break;
                return e;
              default:
                return e;
            }
        } else if (t === "input" && e.type === "hidden") {
          J(o.name, "name");
          var f = o.name == null ? null : "" + o.name;
          if (o.type === "hidden" && e.getAttribute("name") === f)
            return e;
        } else return e;
        if (e = Nl(e.nextSibling), e === null) break;
      }
      return null;
    }
    function Hl(e, t, a) {
      if (t === "") return null;
      for (; e.nodeType !== 3; )
        if ((e.nodeType !== 1 || e.nodeName !== "INPUT" || e.type !== "hidden") && !a || (e = Nl(e.nextSibling), e === null)) return null;
      return e;
    }
    function tu(e) {
      return e.data === mp || e.data === Jc && e.ownerDocument.readyState === Sb;
    }
    function Go(e, t) {
      var a = e.ownerDocument;
      if (e.data !== Jc || a.readyState === Sb)
        t();
      else {
        var i = function() {
          t(), a.removeEventListener("DOMContentLoaded", i);
        };
        a.addEventListener("DOMContentLoaded", i), e._reactRetry = i;
      }
    }
    function Nl(e) {
      for (; e != null; e = e.nextSibling) {
        var t = e.nodeType;
        if (t === 1 || t === 3) break;
        if (t === 8) {
          if (t = e.data, t === cg || t === mp || t === Jc || t === x0 || t === bb)
            break;
          if (t === og) return null;
        }
      }
      return e;
    }
    function _d(e) {
      if (e.nodeType === 1) {
        for (var t = e.nodeName.toLowerCase(), a = {}, i = e.attributes, o = 0; o < i.length; o++) {
          var f = i[o];
          a[tm(f.name)] = f.name.toLowerCase() === "style" ? Mc(e) : f.value;
        }
        return { type: t, props: a };
      }
      return e.nodeType === 8 ? { type: "Suspense", props: {} } : e.nodeValue;
    }
    function Ud(e, t, a) {
      return a === null || a[aT] !== !0 ? (e.nodeValue === t ? e = null : (t = xl(t), e = xl(e.nodeValue) === t ? null : e.nodeValue), e) : null;
    }
    function om(e) {
      e = e.nextSibling;
      for (var t = 0; e; ) {
        if (e.nodeType === 8) {
          var a = e.data;
          if (a === og) {
            if (t === 0)
              return Nl(e.nextSibling);
            t--;
          } else
            a !== cg && a !== mp && a !== Jc || t++;
        }
        e = e.nextSibling;
      }
      return null;
    }
    function Lo(e) {
      e = e.previousSibling;
      for (var t = 0; e; ) {
        if (e.nodeType === 8) {
          var a = e.data;
          if (a === cg || a === mp || a === Jc) {
            if (t === 0) return e;
            t--;
          } else a === og && t++;
        }
        e = e.previousSibling;
      }
      return null;
    }
    function fm(e) {
      Hc(e);
    }
    function _a(e) {
      Hc(e);
    }
    function sm(e, t, a, i, o) {
      switch (o && mr(e, i.ancestorInfo), t = lt(a), e) {
        case "html":
          if (e = t.documentElement, !e)
            throw Error(
              "React expected an <html> element (document.documentElement) to exist in the Document but one was not found. React never removes the documentElement for any Document it renders into so the cause is likely in some other script running on this page."
            );
          return e;
        case "head":
          if (e = t.head, !e)
            throw Error(
              "React expected a <head> element (document.head) to exist in the Document but one was not found. React never removes the head for any Document it renders into so the cause is likely in some other script running on this page."
            );
          return e;
        case "body":
          if (e = t.body, !e)
            throw Error(
              "React expected a <body> element (document.body) to exist in the Document but one was not found. React never removes the body for any Document it renders into so the cause is likely in some other script running on this page."
            );
          return e;
        default:
          throw Error(
            "resolveSingletonInstance was called with an element type that is not supported. This is a bug in React."
          );
      }
    }
    function Ua(e, t, a, i) {
      if (!a[wi] && zl(a)) {
        var o = a.tagName.toLowerCase();
        console.error(
          "You are mounting a new %s component when a previous one has not first unmounted. It is an error to render more than one %s component at a time and attributes and children of these components will likely fail in unpredictable ways. Please only render a single instance of <%s> and if you need to mount a new one, ensure any previous ones have unmounted first.",
          o,
          o,
          o
        );
      }
      switch (e) {
        case "html":
        case "head":
        case "body":
          break;
        default:
          console.error(
            "acquireSingletonInstance was called with an element type that is not supported. This is a bug in React."
          );
      }
      for (o = a.attributes; o.length; )
        a.removeAttributeNode(o[0]);
      kt(a, e, t), a[Zl] = i, a[ga] = t;
    }
    function Vo(e) {
      for (var t = e.attributes; t.length; )
        e.removeAttributeNode(t[0]);
      nn(e);
    }
    function gs(e) {
      return typeof e.getRootNode == "function" ? e.getRootNode() : e.nodeType === 9 ? e : e.ownerDocument;
    }
    function dv(e, t, a) {
      var i = _h;
      if (i && typeof t == "string" && t) {
        var o = Aa(t);
        o = 'link[rel="' + e + '"][href="' + o + '"]', typeof a == "string" && (o += '[crossorigin="' + a + '"]'), Db.has(o) || (Db.add(o), e = { rel: e, crossOrigin: a, href: t }, i.querySelector(o) === null && (t = i.createElement("link"), kt(t, "link", e), D(t), i.head.appendChild(t)));
      }
    }
    function Bu(e, t, a, i) {
      var o = (o = au.current) ? gs(o) : null;
      if (!o)
        throw Error(
          '"resourceRoot" was expected to exist. This is a bug in React.'
        );
      switch (e) {
        case "meta":
        case "title":
          return null;
        case "style":
          return typeof a.precedence == "string" && typeof a.href == "string" ? (a = zi(a.href), t = m(o).hoistableStyles, i = t.get(a), i || (i = {
            type: "style",
            instance: null,
            count: 0,
            state: null
          }, t.set(a, i)), i) : { type: "void", instance: null, count: 0, state: null };
        case "link":
          if (a.rel === "stylesheet" && typeof a.href == "string" && typeof a.precedence == "string") {
            e = zi(a.href);
            var f = m(o).hoistableStyles, d = f.get(e);
            if (!d && (o = o.ownerDocument || o, d = {
              type: "stylesheet",
              instance: null,
              count: 0,
              state: { loading: ar, preload: null }
            }, f.set(e, d), (f = o.querySelector(
              lu(e)
            )) && !f._p && (d.instance = f, d.state.loading = pp | pu), !vu.has(e))) {
              var h = {
                rel: "preload",
                as: "style",
                href: a.href,
                crossOrigin: a.crossOrigin,
                integrity: a.integrity,
                media: a.media,
                hrefLang: a.hrefLang,
                referrerPolicy: a.referrerPolicy
              };
              vu.set(e, h), f || hv(
                o,
                e,
                h,
                d.state
              );
            }
            if (t && i === null)
              throw a = `

  - ` + Uc(t) + `
  + ` + Uc(a), Error(
                "Expected <link> not to update to be updated to a stylesheet with precedence. Check the `rel`, `href`, and `precedence` props of this component. Alternatively, check whether two different <link> components render in the same slot or share the same key." + a
              );
            return d;
          }
          if (t && i !== null)
            throw a = `

  - ` + Uc(t) + `
  + ` + Uc(a), Error(
              "Expected stylesheet with precedence to not be updated to a different kind of <link>. Check the `rel`, `href`, and `precedence` props of this component. Alternatively, check whether two different <link> components render in the same slot or share the same key." + a
            );
          return null;
        case "script":
          return t = a.async, a = a.src, typeof a == "string" && t && typeof t != "function" && typeof t != "symbol" ? (a = Cc(a), t = m(o).hoistableScripts, i = t.get(a), i || (i = {
            type: "script",
            instance: null,
            count: 0,
            state: null
          }, t.set(a, i)), i) : { type: "void", instance: null, count: 0, state: null };
        default:
          throw Error(
            'getResource encountered a type it did not expect: "' + e + '". this is a bug in React.'
          );
      }
    }
    function Uc(e) {
      var t = 0, a = "<link";
      return typeof e.rel == "string" ? (t++, a += ' rel="' + e.rel + '"') : Lu.call(e, "rel") && (t++, a += ' rel="' + (e.rel === null ? "null" : "invalid type " + typeof e.rel) + '"'), typeof e.href == "string" ? (t++, a += ' href="' + e.href + '"') : Lu.call(e, "href") && (t++, a += ' href="' + (e.href === null ? "null" : "invalid type " + typeof e.href) + '"'), typeof e.precedence == "string" ? (t++, a += ' precedence="' + e.precedence + '"') : Lu.call(e, "precedence") && (t++, a += " precedence={" + (e.precedence === null ? "null" : "invalid type " + typeof e.precedence) + "}"), Object.getOwnPropertyNames(e).length > t && (a += " ..."), a + " />";
    }
    function zi(e) {
      return 'href="' + Aa(e) + '"';
    }
    function lu(e) {
      return 'link[rel="stylesheet"][' + e + "]";
    }
    function rm(e) {
      return Je({}, e, {
        "data-precedence": e.precedence,
        precedence: null
      });
    }
    function hv(e, t, a, i) {
      e.querySelector(
        'link[rel="preload"][as="style"][' + t + "]"
      ) ? i.loading = pp : (t = e.createElement("link"), i.preload = t, t.addEventListener("load", function() {
        return i.loading |= pp;
      }), t.addEventListener("error", function() {
        return i.loading |= Rb;
      }), kt(t, "link", a), D(t), e.head.appendChild(t));
    }
    function Cc(e) {
      return '[src="' + Aa(e) + '"]';
    }
    function xc(e) {
      return "script[async]" + e;
    }
    function Cd(e, t, a) {
      if (t.count++, t.instance === null)
        switch (t.type) {
          case "style":
            var i = e.querySelector(
              'style[data-href~="' + Aa(a.href) + '"]'
            );
            if (i)
              return t.instance = i, D(i), i;
            var o = Je({}, a, {
              "data-href": a.href,
              "data-precedence": a.precedence,
              href: null,
              precedence: null
            });
            return i = (e.ownerDocument || e).createElement("style"), D(i), kt(i, "style", o), xd(i, a.precedence, e), t.instance = i;
          case "stylesheet":
            o = zi(a.href);
            var f = e.querySelector(
              lu(o)
            );
            if (f)
              return t.state.loading |= pu, t.instance = f, D(f), f;
            i = rm(a), (o = vu.get(o)) && dm(i, o), f = (e.ownerDocument || e).createElement("link"), D(f);
            var d = f;
            return d._p = new Promise(function(h, v) {
              d.onload = h, d.onerror = v;
            }), kt(f, "link", i), t.state.loading |= pu, xd(f, a.precedence, e), t.instance = f;
          case "script":
            return f = Cc(a.src), (o = e.querySelector(
              xc(f)
            )) ? (t.instance = o, D(o), o) : (i = a, (o = vu.get(f)) && (i = Je({}, a), hm(i, o)), e = e.ownerDocument || e, o = e.createElement("script"), D(o), kt(o, "link", i), e.head.appendChild(o), t.instance = o);
          case "void":
            return null;
          default:
            throw Error(
              'acquireResource encountered a resource type it did not expect: "' + t.type + '". this is a bug in React.'
            );
        }
      else
        t.type === "stylesheet" && (t.state.loading & pu) === ar && (i = t.instance, t.state.loading |= pu, xd(i, a.precedence, e));
      return t.instance;
    }
    function xd(e, t, a) {
      for (var i = a.querySelectorAll(
        'link[rel="stylesheet"][data-precedence],style[data-precedence]'
      ), o = i.length ? i[i.length - 1] : null, f = o, d = 0; d < i.length; d++) {
        var h = i[d];
        if (h.dataset.precedence === t) f = h;
        else if (f !== o) break;
      }
      f ? f.parentNode.insertBefore(e, f.nextSibling) : (t = a.nodeType === 9 ? a.head : a, t.insertBefore(e, t.firstChild));
    }
    function dm(e, t) {
      e.crossOrigin == null && (e.crossOrigin = t.crossOrigin), e.referrerPolicy == null && (e.referrerPolicy = t.referrerPolicy), e.title == null && (e.title = t.title);
    }
    function hm(e, t) {
      e.crossOrigin == null && (e.crossOrigin = t.crossOrigin), e.referrerPolicy == null && (e.referrerPolicy = t.referrerPolicy), e.integrity == null && (e.integrity = t.integrity);
    }
    function ym(e, t, a) {
      if (sg === null) {
        var i = /* @__PURE__ */ new Map(), o = sg = /* @__PURE__ */ new Map();
        o.set(a, i);
      } else
        o = sg, i = o.get(a), i || (i = /* @__PURE__ */ new Map(), o.set(a, i));
      if (i.has(e)) return i;
      for (i.set(e, null), a = a.getElementsByTagName(e), o = 0; o < a.length; o++) {
        var f = a[o];
        if (!(f[Po] || f[Zl] || e === "link" && f.getAttribute("rel") === "stylesheet") && f.namespaceURI !== af) {
          var d = f.getAttribute(t) || "";
          d = e + d;
          var h = i.get(d);
          h ? h.push(f) : i.set(d, [f]);
        }
      }
      return i;
    }
    function mm(e, t, a) {
      e = e.ownerDocument || e, e.head.insertBefore(
        a,
        t === "title" ? e.querySelector("head > title") : null
      );
    }
    function Xo(e, t, a) {
      var i = !a.ancestorInfo.containerTagInScope;
      if (a.context === Mh || t.itemProp != null)
        return !i || t.itemProp == null || e !== "meta" && e !== "title" && e !== "style" && e !== "link" && e !== "script" || console.error(
          "Cannot render a <%s> outside the main document if it has an `itemProp` prop. `itemProp` suggests the tag belongs to an `itemScope` which can appear anywhere in the DOM. If you were intending for React to hoist this <%s> remove the `itemProp` prop. Otherwise, try moving this tag into the <head> or <body> of the Document.",
          e,
          e
        ), !1;
      switch (e) {
        case "meta":
        case "title":
          return !0;
        case "style":
          if (typeof t.precedence != "string" || typeof t.href != "string" || t.href === "") {
            i && console.error(
              'Cannot render a <style> outside the main document without knowing its precedence and a unique href key. React can hoist and deduplicate <style> tags if you provide a `precedence` prop along with an `href` prop that does not conflict with the `href` values used in any other hoisted <style> or <link rel="stylesheet" ...> tags.  Note that hoisting <style> tags is considered an advanced feature that most will not use directly. Consider moving the <style> tag to the <head> or consider adding a `precedence="default"` and `href="some unique resource identifier"`.'
            );
            break;
          }
          return !0;
        case "link":
          if (typeof t.rel != "string" || typeof t.href != "string" || t.href === "" || t.onLoad || t.onError) {
            if (t.rel === "stylesheet" && typeof t.precedence == "string") {
              e = t.href;
              var o = t.onError, f = t.disabled;
              a = [], t.onLoad && a.push("`onLoad`"), o && a.push("`onError`"), f != null && a.push("`disabled`"), o = tt(a, "and"), o += a.length === 1 ? " prop" : " props", f = a.length === 1 ? "an " + o : "the " + o, a.length && console.error(
                'React encountered a <link rel="stylesheet" href="%s" ... /> with a `precedence` prop that also included %s. The presence of loading and error handlers indicates an intent to manage the stylesheet loading state from your from your Component code and React will not hoist or deduplicate this stylesheet. If your intent was to have React hoist and deduplciate this stylesheet using the `precedence` prop remove the %s, otherwise remove the `precedence` prop.',
                e,
                f,
                o
              );
            }
            i && (typeof t.rel != "string" || typeof t.href != "string" || t.href === "" ? console.error(
              "Cannot render a <link> outside the main document without a `rel` and `href` prop. Try adding a `rel` and/or `href` prop to this <link> or moving the link into the <head> tag"
            ) : (t.onError || t.onLoad) && console.error(
              "Cannot render a <link> with onLoad or onError listeners outside the main document. Try removing onLoad={...} and onError={...} or moving it into the root <head> tag or somewhere in the <body>."
            ));
            break;
          }
          switch (t.rel) {
            case "stylesheet":
              return e = t.precedence, t = t.disabled, typeof e != "string" && i && console.error(
                'Cannot render a <link rel="stylesheet" /> outside the main document without knowing its precedence. Consider adding precedence="default" or moving it into the root <head> tag.'
              ), typeof e == "string" && t == null;
            default:
              return !0;
          }
        case "script":
          if (e = t.async && typeof t.async != "function" && typeof t.async != "symbol", !e || t.onLoad || t.onError || !t.src || typeof t.src != "string") {
            i && (e ? t.onLoad || t.onError ? console.error(
              "Cannot render a <script> with onLoad or onError listeners outside the main document. Try removing onLoad={...} and onError={...} or moving it into the root <head> tag or somewhere in the <body>."
            ) : console.error(
              "Cannot render a <script> outside the main document without `async={true}` and a non-empty `src` prop. Ensure there is a valid `src` and either make the script async or move it into the root <head> tag or somewhere in the <body>."
            ) : console.error(
              'Cannot render a sync or defer <script> outside the main document without knowing its order. Try adding async="" or moving it into the root <head> tag.'
            ));
            break;
          }
          return !0;
        case "noscript":
        case "template":
          i && console.error(
            "Cannot render <%s> outside the main document. Try moving it into the root <head> tag.",
            e
          );
      }
      return !1;
    }
    function bs(e) {
      return !(e.type === "stylesheet" && (e.state.loading & Ob) === ar);
    }
    function yv() {
    }
    function mv(e, t, a) {
      if (vp === null)
        throw Error(
          "Internal React Error: suspendedState null when it was expected to exists. Please report this as a React bug."
        );
      var i = vp;
      if (t.type === "stylesheet" && (typeof a.media != "string" || matchMedia(a.media).matches !== !1) && (t.state.loading & pu) === ar) {
        if (t.instance === null) {
          var o = zi(a.href), f = e.querySelector(
            lu(o)
          );
          if (f) {
            e = f._p, e !== null && typeof e == "object" && typeof e.then == "function" && (i.count++, i = Ss.bind(i), e.then(i, i)), t.state.loading |= pu, t.instance = f, D(f);
            return;
          }
          f = e.ownerDocument || e, a = rm(a), (o = vu.get(o)) && dm(a, o), f = f.createElement("link"), D(f);
          var d = f;
          d._p = new Promise(function(h, v) {
            d.onload = h, d.onerror = v;
          }), kt(f, "link", a), t.instance = f;
        }
        i.stylesheets === null && (i.stylesheets = /* @__PURE__ */ new Map()), i.stylesheets.set(t, e), (e = t.state.preload) && (t.state.loading & Ob) === ar && (i.count++, t = Ss.bind(i), e.addEventListener("load", t), e.addEventListener("error", t));
      }
    }
    function pv() {
      if (vp === null)
        throw Error(
          "Internal React Error: suspendedState null when it was expected to exists. Please report this as a React bug."
        );
      var e = vp;
      return e.stylesheets && e.count === 0 && Hd(e, e.stylesheets), 0 < e.count ? function(t) {
        var a = setTimeout(function() {
          if (e.stylesheets && Hd(e, e.stylesheets), e.unsuspend) {
            var i = e.unsuspend;
            e.unsuspend = null, i();
          }
        }, 6e4);
        return e.unsuspend = t, function() {
          e.unsuspend = null, clearTimeout(a);
        };
      } : null;
    }
    function Ss() {
      if (this.count--, this.count === 0) {
        if (this.stylesheets)
          Hd(this, this.stylesheets);
        else if (this.unsuspend) {
          var e = this.unsuspend;
          this.unsuspend = null, e();
        }
      }
    }
    function Hd(e, t) {
      e.stylesheets = null, e.unsuspend !== null && (e.count++, rg = /* @__PURE__ */ new Map(), t.forEach(vv, e), rg = null, Ss.call(e));
    }
    function vv(e, t) {
      if (!(t.state.loading & pu)) {
        var a = rg.get(e);
        if (a) var i = a.get(j0);
        else {
          a = /* @__PURE__ */ new Map(), rg.set(e, a);
          for (var o = e.querySelectorAll(
            "link[data-precedence],style[data-precedence]"
          ), f = 0; f < o.length; f++) {
            var d = o[f];
            (d.nodeName === "LINK" || d.getAttribute("media") !== "not all") && (a.set(d.dataset.precedence, d), i = d);
          }
          i && a.set(j0, i);
        }
        o = t.instance, d = o.getAttribute("data-precedence"), f = a.get(d) || i, f === i && a.set(j0, o), a.set(d, o), this.count++, i = Ss.bind(this), o.addEventListener("load", i), o.addEventListener("error", i), f ? f.parentNode.insertBefore(o, f.nextSibling) : (e = e.nodeType === 9 ? e.head : e, e.insertBefore(o, e.firstChild)), t.state.loading |= pu;
      }
    }
    function Nd(e, t, a, i, o, f, d, h) {
      for (this.tag = 1, this.containerInfo = e, this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = lr, this.callbackNode = this.next = this.pendingContext = this.context = this.cancelPendingCommit = null, this.callbackPriority = 0, this.expirationTimes = eo(-1), this.entangledLanes = this.shellSuspendCounter = this.errorRecoveryDisabledLanes = this.expiredLanes = this.warmLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = eo(0), this.hiddenUpdates = eo(null), this.identifierPrefix = i, this.onUncaughtError = o, this.onCaughtError = f, this.onRecoverableError = d, this.pooledCache = null, this.pooledCacheLanes = 0, this.formState = h, this.incompleteTransitions = /* @__PURE__ */ new Map(), this.passiveEffectDuration = this.effectDuration = -0, this.memoizedUpdaters = /* @__PURE__ */ new Set(), e = this.pendingUpdatersLaneMap = [], t = 0; 31 > t; t++) e.push(/* @__PURE__ */ new Set());
      this._debugRootType = a ? "hydrateRoot()" : "createRoot()";
    }
    function pm(e, t, a, i, o, f, d, h, v, b, B, X) {
      return e = new Nd(
        e,
        t,
        a,
        d,
        h,
        v,
        b,
        X
      ), t = HS, f === !0 && (t |= Sa | Ku), Ft && (t |= ta), f = z(3, null, null, t), e.current = f, f.stateNode = e, t = Bf(), cc(t), e.pooledCache = t, cc(t), f.memoizedState = {
        element: i,
        isDehydrated: a,
        cache: t
      }, ca(f), e;
    }
    function vm(e) {
      return e ? (e = nf, e) : nf;
    }
    function Tt(e, t, a, i, o, f) {
      if (wl && typeof wl.onScheduleFiberRoot == "function")
        try {
          wl.onScheduleFiberRoot(Hi, i, a);
        } catch (d) {
          va || (va = !0, console.error(
            "React instrumentation encountered an error: %s",
            d
          ));
        }
      fe !== null && typeof fe.markRenderScheduled == "function" && fe.markRenderScheduled(t), o = vm(o), i.context === null ? i.context = o : i.pendingContext = o, ba && xa !== null && !Ub && (Ub = !0, console.error(
        `Render methods should be a pure function of props and state; triggering nested component updates from render is not allowed. If necessary, trigger nested updates in componentDidUpdate.

Check the render method of %s.`,
        re(xa) || "Unknown"
      )), i = wn(t), i.payload = { element: a }, f = f === void 0 ? null : f, f !== null && (typeof f != "function" && console.error(
        "Expected the last optional `callback` argument to be a function. Instead received: %s.",
        f
      ), i.callback = f), a = hn(e, i, t), a !== null && (Kt(a, e, t), di(a, e, t));
    }
    function wd(e, t) {
      if (e = e.memoizedState, e !== null && e.dehydrated !== null) {
        var a = e.retryLane;
        e.retryLane = a !== 0 && a < t ? a : t;
      }
    }
    function gm(e, t) {
      wd(e, t), (e = e.alternate) && wd(e, t);
    }
    function bm(e) {
      if (e.tag === 13) {
        var t = ia(e, 67108864);
        t !== null && Kt(t, e, 67108864), gm(e, 67108864);
      }
    }
    function Cg() {
      return xa;
    }
    function xg() {
      for (var e = /* @__PURE__ */ new Map(), t = 1, a = 0; 31 > a; a++) {
        var i = bf(t);
        e.set(t, i), t *= 2;
      }
      return e;
    }
    function Hg(e, t, a, i) {
      var o = L.T;
      L.T = null;
      var f = Ce.p;
      try {
        Ce.p = ql, Mi(e, t, a, i);
      } finally {
        Ce.p = f, L.T = o;
      }
    }
    function qd(e, t, a, i) {
      var o = L.T;
      L.T = null;
      var f = Ce.p;
      try {
        Ce.p = En, Mi(e, t, a, i);
      } finally {
        Ce.p = f, L.T = o;
      }
    }
    function Mi(e, t, a, i) {
      if (hg) {
        var o = Ts(i);
        if (o === null)
          Fl(
            e,
            t,
            i,
            yg,
            a
          ), _i(e, i);
        else if (Es(
          o,
          e,
          t,
          a,
          i
        ))
          i.stopPropagation();
        else if (_i(e, i), t & 4 && -1 < rT.indexOf(e)) {
          for (; o !== null; ) {
            var f = zl(o);
            if (f !== null)
              switch (f.tag) {
                case 3:
                  if (f = f.stateNode, f.current.memoizedState.isDehydrated) {
                    var d = tl(f.pendingLanes);
                    if (d !== 0) {
                      var h = f;
                      for (h.pendingLanes |= 2, h.entangledLanes |= 2; d; ) {
                        var v = 1 << 31 - Ql(d);
                        h.entanglements[1] |= v, d &= ~v;
                      }
                      $a(f), (Et & (qa | Wu)) === An && (Fv = nu() + ob, Dc(0));
                    }
                  }
                  break;
                case 13:
                  h = ia(f, 2), h !== null && Kt(h, f, 2), Rc(), gm(f, 2);
              }
            if (f = Ts(i), f === null && Fl(
              e,
              t,
              i,
              yg,
              a
            ), f === o) break;
            o = f;
          }
          o !== null && i.stopPropagation();
        } else
          Fl(
            e,
            t,
            i,
            null,
            a
          );
      }
    }
    function Ts(e) {
      return e = Pi(e), Qo(e);
    }
    function Qo(e) {
      if (yg = null, e = ua(e), e !== null) {
        var t = nt(e);
        if (t === null) e = null;
        else {
          var a = t.tag;
          if (a === 13) {
            if (e = el(t), e !== null) return e;
            e = null;
          } else if (a === 3) {
            if (t.stateNode.current.memoizedState.isDehydrated)
              return t.tag === 3 ? t.stateNode.containerInfo : null;
            e = null;
          } else t !== e && (e = null);
        }
      }
      return yg = e, null;
    }
    function jd(e) {
      switch (e) {
        case "beforetoggle":
        case "cancel":
        case "click":
        case "close":
        case "contextmenu":
        case "copy":
        case "cut":
        case "auxclick":
        case "dblclick":
        case "dragend":
        case "dragstart":
        case "drop":
        case "focusin":
        case "focusout":
        case "input":
        case "invalid":
        case "keydown":
        case "keypress":
        case "keyup":
        case "mousedown":
        case "mouseup":
        case "paste":
        case "pause":
        case "play":
        case "pointercancel":
        case "pointerdown":
        case "pointerup":
        case "ratechange":
        case "reset":
        case "resize":
        case "seeked":
        case "submit":
        case "toggle":
        case "touchcancel":
        case "touchend":
        case "touchstart":
        case "volumechange":
        case "change":
        case "selectionchange":
        case "textInput":
        case "compositionstart":
        case "compositionend":
        case "compositionupdate":
        case "beforeblur":
        case "afterblur":
        case "beforeinput":
        case "blur":
        case "fullscreenchange":
        case "focus":
        case "hashchange":
        case "popstate":
        case "select":
        case "selectstart":
          return ql;
        case "drag":
        case "dragenter":
        case "dragexit":
        case "dragleave":
        case "dragover":
        case "mousemove":
        case "mouseout":
        case "mouseover":
        case "pointermove":
        case "pointerout":
        case "pointerover":
        case "scroll":
        case "touchmove":
        case "wheel":
        case "mouseenter":
        case "mouseleave":
        case "pointerenter":
        case "pointerleave":
          return En;
        case "message":
          switch (xi()) {
            case Vd:
              return ql;
            case Cs:
              return En;
            case Wo:
            case Bg:
              return Xu;
            case xs:
              return Kd;
            default:
              return Xu;
          }
        default:
          return Xu;
      }
    }
    function _i(e, t) {
      switch (e) {
        case "focusin":
        case "focusout":
          mf = null;
          break;
        case "dragenter":
        case "dragleave":
          pf = null;
          break;
        case "mouseover":
        case "mouseout":
          vf = null;
          break;
        case "pointerover":
        case "pointerout":
          bp.delete(t.pointerId);
          break;
        case "gotpointercapture":
        case "lostpointercapture":
          Sp.delete(t.pointerId);
      }
    }
    function pa(e, t, a, i, o, f) {
      return e === null || e.nativeEvent !== f ? (e = {
        blockedOn: t,
        domEventName: a,
        eventSystemFlags: i,
        nativeEvent: f,
        targetContainers: [o]
      }, t !== null && (t = zl(t), t !== null && bm(t)), e) : (e.eventSystemFlags |= i, t = e.targetContainers, o !== null && t.indexOf(o) === -1 && t.push(o), e);
    }
    function Es(e, t, a, i, o) {
      switch (t) {
        case "focusin":
          return mf = pa(
            mf,
            e,
            t,
            a,
            i,
            o
          ), !0;
        case "dragenter":
          return pf = pa(
            pf,
            e,
            t,
            a,
            i,
            o
          ), !0;
        case "mouseover":
          return vf = pa(
            vf,
            e,
            t,
            a,
            i,
            o
          ), !0;
        case "pointerover":
          var f = o.pointerId;
          return bp.set(
            f,
            pa(
              bp.get(f) || null,
              e,
              t,
              a,
              i,
              o
            )
          ), !0;
        case "gotpointercapture":
          return f = o.pointerId, Sp.set(
            f,
            pa(
              Sp.get(f) || null,
              e,
              t,
              a,
              i,
              o
            )
          ), !0;
      }
      return !1;
    }
    function gv(e) {
      var t = ua(e.target);
      if (t !== null) {
        var a = nt(t);
        if (a !== null) {
          if (t = a.tag, t === 13) {
            if (t = el(a), t !== null) {
              e.blockedOn = t, lo(e.priority, function() {
                if (a.tag === 13) {
                  var i = ha(a);
                  i = Ol(i);
                  var o = ia(
                    a,
                    i
                  );
                  o !== null && Kt(o, a, i), gm(a, i);
                }
              });
              return;
            }
          } else if (t === 3 && a.stateNode.current.memoizedState.isDehydrated) {
            e.blockedOn = a.tag === 3 ? a.stateNode.containerInfo : null;
            return;
          }
        }
      }
      e.blockedOn = null;
    }
    function As(e) {
      if (e.blockedOn !== null) return !1;
      for (var t = e.targetContainers; 0 < t.length; ) {
        var a = Ts(e.nativeEvent);
        if (a === null) {
          a = e.nativeEvent;
          var i = new a.constructor(
            a.type,
            a
          ), o = i;
          s !== null && console.error(
            "Expected currently replaying event to be null. This error is likely caused by a bug in React. Please file an issue."
          ), s = o, a.target.dispatchEvent(i), s === null && console.error(
            "Expected currently replaying event to not be null. This error is likely caused by a bug in React. Please file an issue."
          ), s = null;
        } else
          return t = zl(a), t !== null && bm(t), e.blockedOn = a, !1;
        t.shift();
      }
      return !0;
    }
    function Sm(e, t, a) {
      As(e) && a.delete(t);
    }
    function bv() {
      B0 = !1, mf !== null && As(mf) && (mf = null), pf !== null && As(pf) && (pf = null), vf !== null && As(vf) && (vf = null), bp.forEach(Sm), Sp.forEach(Sm);
    }
    function Rs(e, t) {
      e.blockedOn === t && (e.blockedOn = null, B0 || (B0 = !0, Wt.unstable_scheduleCallback(
        Wt.unstable_NormalPriority,
        bv
      )));
    }
    function Sv(e) {
      mg !== e && (mg = e, Wt.unstable_scheduleCallback(
        Wt.unstable_NormalPriority,
        function() {
          mg === e && (mg = null);
          for (var t = 0; t < e.length; t += 3) {
            var a = e[t], i = e[t + 1], o = e[t + 2];
            if (typeof i != "function") {
              if (Qo(i || a) === null)
                continue;
              break;
            }
            var f = zl(a);
            f !== null && (e.splice(t, 3), t -= 3, a = {
              pending: !0,
              data: o,
              method: a.method,
              action: i
            }, Object.freeze(a), hc(
              f,
              a,
              i,
              o
            ));
          }
        }
      ));
    }
    function Hc(e) {
      function t(v) {
        return Rs(v, e);
      }
      mf !== null && Rs(mf, e), pf !== null && Rs(pf, e), vf !== null && Rs(vf, e), bp.forEach(t), Sp.forEach(t);
      for (var a = 0; a < gf.length; a++) {
        var i = gf[a];
        i.blockedOn === e && (i.blockedOn = null);
      }
      for (; 0 < gf.length && (a = gf[0], a.blockedOn === null); )
        gv(a), a.blockedOn === null && gf.shift();
      if (a = (e.ownerDocument || e).$$reactFormReplay, a != null)
        for (i = 0; i < a.length; i += 3) {
          var o = a[i], f = a[i + 1], d = o[ga] || null;
          if (typeof f == "function")
            d || Sv(a);
          else if (d) {
            var h = null;
            if (f && f.hasAttribute("formAction")) {
              if (o = f, d = f[ga] || null)
                h = d.formAction;
              else if (Qo(o) !== null) continue;
            } else h = d.action;
            typeof h == "function" ? a[i + 1] = h : (a.splice(i, 3), i -= 3), Sv(a);
          }
        }
    }
    function Bd(e) {
      this._internalRoot = e;
    }
    function Os(e) {
      this._internalRoot = e;
    }
    function Tv(e) {
      e[wi] && (e._reactRootContainer ? console.error(
        "You are calling ReactDOMClient.createRoot() on a container that was previously passed to ReactDOM.render(). This is not supported."
      ) : console.error(
        "You are calling ReactDOMClient.createRoot() on a container that has already been passed to createRoot() before. Instead, call root.render() on the existing root instead if you want to update it."
      ));
    }
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var Wt = iS(), Ds = Ch(), Ng = cS(), Je = Object.assign, zs = Symbol.for("react.element"), Ui = Symbol.for("react.transitional.element"), Nc = Symbol.for("react.portal"), Le = Symbol.for("react.fragment"), Zo = Symbol.for("react.strict_mode"), Ko = Symbol.for("react.profiler"), Tm = Symbol.for("react.provider"), Yd = Symbol.for("react.consumer"), Ia = Symbol.for("react.context"), Yu = Symbol.for("react.forward_ref"), Jo = Symbol.for("react.suspense"), Ci = Symbol.for("react.suspense_list"), Ms = Symbol.for("react.memo"), Ca = Symbol.for("react.lazy"), Em = Symbol.for("react.activity"), Ev = Symbol.for("react.memo_cache_sentinel"), Am = Symbol.iterator, Gd = Symbol.for("react.client.reference"), we = Array.isArray, L = Ds.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, Ce = Ng.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, wg = Object.freeze({
      pending: !1,
      data: null,
      method: null,
      action: null
    }), _s = [], Us = [], Pa = -1, Gu = Rt(null), ko = Rt(null), au = Rt(null), $o = Rt(null), Lu = Object.prototype.hasOwnProperty, Ld = Wt.unstable_scheduleCallback, qg = Wt.unstable_cancelCallback, Av = Wt.unstable_shouldYield, jg = Wt.unstable_requestPaint, nu = Wt.unstable_now, xi = Wt.unstable_getCurrentPriorityLevel, Vd = Wt.unstable_ImmediatePriority, Cs = Wt.unstable_UserBlockingPriority, Wo = Wt.unstable_NormalPriority, Bg = Wt.unstable_LowPriority, xs = Wt.unstable_IdlePriority, Yg = Wt.log, Tn = Wt.unstable_setDisableYieldValue, Hi = null, wl = null, fe = null, va = !1, Ft = typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u", Ql = Math.clz32 ? Math.clz32 : Pc, Xd = Math.log, Vu = Math.LN2, Qd = 256, Zd = 4194304, ql = 2, En = 8, Xu = 32, Kd = 268435456, Ni = Math.random().toString(36).slice(2), Zl = "__reactFiber$" + Ni, ga = "__reactProps$" + Ni, wi = "__reactContainer$" + Ni, Rm = "__reactEvents$" + Ni, Rv = "__reactListeners$" + Ni, Fo = "__reactHandles$" + Ni, Io = "__reactResources$" + Ni, Po = "__reactMarker$" + Ni, Ov = /* @__PURE__ */ new Set(), en = {}, wc = {}, Dv = {
      button: !0,
      checkbox: !0,
      image: !0,
      hidden: !0,
      radio: !0,
      reset: !0,
      submit: !0
    }, Jd = RegExp(
      "^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), kd = {}, $d = {}, qi = 0, Om, Dm, zv, zm, ef, Mv, _v;
    cn.__reactDisabledLog = !0;
    var Mm, Hs, tf = !1, Ns = new (typeof WeakMap == "function" ? WeakMap : Map)(), xa = null, ba = !1, Gg = /[\n"\\]/g, _m = !1, Um = !1, Cm = !1, xm = !1, Wd = !1, Hm = !1, ws = ["value", "defaultValue"], Uv = !1, Cv = /["'&<>\n\t]|^\s|\s$/, Nm = "address applet area article aside base basefont bgsound blockquote body br button caption center col colgroup dd details dir div dl dt embed fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hgroup hr html iframe img input isindex li link listing main marquee menu menuitem meta nav noembed noframes noscript object ol p param plaintext pre script section select source style summary table tbody td template textarea tfoot th thead title tr track ul wbr xmp".split(
      " "
    ), Fd = "applet caption html table td th marquee object template foreignObject desc title".split(
      " "
    ), Id = Fd.concat(["button"]), wm = "dd dt li option optgroup p rp rt".split(" "), qm = {
      current: null,
      formTag: null,
      aTagInScope: null,
      buttonTagInScope: null,
      nobrTagInScope: null,
      pTagInButtonScope: null,
      listItemTagAutoclosing: null,
      dlItemTagAutoclosing: null,
      containerTagInScope: null,
      implicitRootScope: !1
    }, lf = {}, uu = {
      animation: "animationDelay animationDirection animationDuration animationFillMode animationIterationCount animationName animationPlayState animationTimingFunction".split(
        " "
      ),
      background: "backgroundAttachment backgroundClip backgroundColor backgroundImage backgroundOrigin backgroundPositionX backgroundPositionY backgroundRepeat backgroundSize".split(
        " "
      ),
      backgroundPosition: ["backgroundPositionX", "backgroundPositionY"],
      border: "borderBottomColor borderBottomStyle borderBottomWidth borderImageOutset borderImageRepeat borderImageSlice borderImageSource borderImageWidth borderLeftColor borderLeftStyle borderLeftWidth borderRightColor borderRightStyle borderRightWidth borderTopColor borderTopStyle borderTopWidth".split(
        " "
      ),
      borderBlockEnd: [
        "borderBlockEndColor",
        "borderBlockEndStyle",
        "borderBlockEndWidth"
      ],
      borderBlockStart: [
        "borderBlockStartColor",
        "borderBlockStartStyle",
        "borderBlockStartWidth"
      ],
      borderBottom: [
        "borderBottomColor",
        "borderBottomStyle",
        "borderBottomWidth"
      ],
      borderColor: [
        "borderBottomColor",
        "borderLeftColor",
        "borderRightColor",
        "borderTopColor"
      ],
      borderImage: [
        "borderImageOutset",
        "borderImageRepeat",
        "borderImageSlice",
        "borderImageSource",
        "borderImageWidth"
      ],
      borderInlineEnd: [
        "borderInlineEndColor",
        "borderInlineEndStyle",
        "borderInlineEndWidth"
      ],
      borderInlineStart: [
        "borderInlineStartColor",
        "borderInlineStartStyle",
        "borderInlineStartWidth"
      ],
      borderLeft: ["borderLeftColor", "borderLeftStyle", "borderLeftWidth"],
      borderRadius: [
        "borderBottomLeftRadius",
        "borderBottomRightRadius",
        "borderTopLeftRadius",
        "borderTopRightRadius"
      ],
      borderRight: [
        "borderRightColor",
        "borderRightStyle",
        "borderRightWidth"
      ],
      borderStyle: [
        "borderBottomStyle",
        "borderLeftStyle",
        "borderRightStyle",
        "borderTopStyle"
      ],
      borderTop: ["borderTopColor", "borderTopStyle", "borderTopWidth"],
      borderWidth: [
        "borderBottomWidth",
        "borderLeftWidth",
        "borderRightWidth",
        "borderTopWidth"
      ],
      columnRule: ["columnRuleColor", "columnRuleStyle", "columnRuleWidth"],
      columns: ["columnCount", "columnWidth"],
      flex: ["flexBasis", "flexGrow", "flexShrink"],
      flexFlow: ["flexDirection", "flexWrap"],
      font: "fontFamily fontFeatureSettings fontKerning fontLanguageOverride fontSize fontSizeAdjust fontStretch fontStyle fontVariant fontVariantAlternates fontVariantCaps fontVariantEastAsian fontVariantLigatures fontVariantNumeric fontVariantPosition fontWeight lineHeight".split(
        " "
      ),
      fontVariant: "fontVariantAlternates fontVariantCaps fontVariantEastAsian fontVariantLigatures fontVariantNumeric fontVariantPosition".split(
        " "
      ),
      gap: ["columnGap", "rowGap"],
      grid: "gridAutoColumns gridAutoFlow gridAutoRows gridTemplateAreas gridTemplateColumns gridTemplateRows".split(
        " "
      ),
      gridArea: [
        "gridColumnEnd",
        "gridColumnStart",
        "gridRowEnd",
        "gridRowStart"
      ],
      gridColumn: ["gridColumnEnd", "gridColumnStart"],
      gridColumnGap: ["columnGap"],
      gridGap: ["columnGap", "rowGap"],
      gridRow: ["gridRowEnd", "gridRowStart"],
      gridRowGap: ["rowGap"],
      gridTemplate: [
        "gridTemplateAreas",
        "gridTemplateColumns",
        "gridTemplateRows"
      ],
      listStyle: ["listStyleImage", "listStylePosition", "listStyleType"],
      margin: ["marginBottom", "marginLeft", "marginRight", "marginTop"],
      marker: ["markerEnd", "markerMid", "markerStart"],
      mask: "maskClip maskComposite maskImage maskMode maskOrigin maskPositionX maskPositionY maskRepeat maskSize".split(
        " "
      ),
      maskPosition: ["maskPositionX", "maskPositionY"],
      outline: ["outlineColor", "outlineStyle", "outlineWidth"],
      overflow: ["overflowX", "overflowY"],
      padding: ["paddingBottom", "paddingLeft", "paddingRight", "paddingTop"],
      placeContent: ["alignContent", "justifyContent"],
      placeItems: ["alignItems", "justifyItems"],
      placeSelf: ["alignSelf", "justifySelf"],
      textDecoration: [
        "textDecorationColor",
        "textDecorationLine",
        "textDecorationStyle"
      ],
      textEmphasis: ["textEmphasisColor", "textEmphasisStyle"],
      transition: [
        "transitionDelay",
        "transitionDuration",
        "transitionProperty",
        "transitionTimingFunction"
      ],
      wordWrap: ["overflowWrap"]
    }, Qu = /([A-Z])/g, Zu = /^ms-/, qs = /^(?:webkit|moz|o)[A-Z]/, js = /^-ms-/, ji = /-(.)/g, xv = /;\s*$/, qc = {}, jc = {}, Hv = !1, jm = !1, Bs = new Set(
      "animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(
        " "
      )
    ), Ys = "http://www.w3.org/1998/Math/MathML", af = "http://www.w3.org/2000/svg", Pd = /* @__PURE__ */ new Map([
      ["acceptCharset", "accept-charset"],
      ["htmlFor", "for"],
      ["httpEquiv", "http-equiv"],
      ["crossOrigin", "crossorigin"],
      ["accentHeight", "accent-height"],
      ["alignmentBaseline", "alignment-baseline"],
      ["arabicForm", "arabic-form"],
      ["baselineShift", "baseline-shift"],
      ["capHeight", "cap-height"],
      ["clipPath", "clip-path"],
      ["clipRule", "clip-rule"],
      ["colorInterpolation", "color-interpolation"],
      ["colorInterpolationFilters", "color-interpolation-filters"],
      ["colorProfile", "color-profile"],
      ["colorRendering", "color-rendering"],
      ["dominantBaseline", "dominant-baseline"],
      ["enableBackground", "enable-background"],
      ["fillOpacity", "fill-opacity"],
      ["fillRule", "fill-rule"],
      ["floodColor", "flood-color"],
      ["floodOpacity", "flood-opacity"],
      ["fontFamily", "font-family"],
      ["fontSize", "font-size"],
      ["fontSizeAdjust", "font-size-adjust"],
      ["fontStretch", "font-stretch"],
      ["fontStyle", "font-style"],
      ["fontVariant", "font-variant"],
      ["fontWeight", "font-weight"],
      ["glyphName", "glyph-name"],
      ["glyphOrientationHorizontal", "glyph-orientation-horizontal"],
      ["glyphOrientationVertical", "glyph-orientation-vertical"],
      ["horizAdvX", "horiz-adv-x"],
      ["horizOriginX", "horiz-origin-x"],
      ["imageRendering", "image-rendering"],
      ["letterSpacing", "letter-spacing"],
      ["lightingColor", "lighting-color"],
      ["markerEnd", "marker-end"],
      ["markerMid", "marker-mid"],
      ["markerStart", "marker-start"],
      ["overlinePosition", "overline-position"],
      ["overlineThickness", "overline-thickness"],
      ["paintOrder", "paint-order"],
      ["panose-1", "panose-1"],
      ["pointerEvents", "pointer-events"],
      ["renderingIntent", "rendering-intent"],
      ["shapeRendering", "shape-rendering"],
      ["stopColor", "stop-color"],
      ["stopOpacity", "stop-opacity"],
      ["strikethroughPosition", "strikethrough-position"],
      ["strikethroughThickness", "strikethrough-thickness"],
      ["strokeDasharray", "stroke-dasharray"],
      ["strokeDashoffset", "stroke-dashoffset"],
      ["strokeLinecap", "stroke-linecap"],
      ["strokeLinejoin", "stroke-linejoin"],
      ["strokeMiterlimit", "stroke-miterlimit"],
      ["strokeOpacity", "stroke-opacity"],
      ["strokeWidth", "stroke-width"],
      ["textAnchor", "text-anchor"],
      ["textDecoration", "text-decoration"],
      ["textRendering", "text-rendering"],
      ["transformOrigin", "transform-origin"],
      ["underlinePosition", "underline-position"],
      ["underlineThickness", "underline-thickness"],
      ["unicodeBidi", "unicode-bidi"],
      ["unicodeRange", "unicode-range"],
      ["unitsPerEm", "units-per-em"],
      ["vAlphabetic", "v-alphabetic"],
      ["vHanging", "v-hanging"],
      ["vIdeographic", "v-ideographic"],
      ["vMathematical", "v-mathematical"],
      ["vectorEffect", "vector-effect"],
      ["vertAdvY", "vert-adv-y"],
      ["vertOriginX", "vert-origin-x"],
      ["vertOriginY", "vert-origin-y"],
      ["wordSpacing", "word-spacing"],
      ["writingMode", "writing-mode"],
      ["xmlnsXlink", "xmlns:xlink"],
      ["xHeight", "x-height"]
    ]), Bc = {
      accept: "accept",
      acceptcharset: "acceptCharset",
      "accept-charset": "acceptCharset",
      accesskey: "accessKey",
      action: "action",
      allowfullscreen: "allowFullScreen",
      alt: "alt",
      as: "as",
      async: "async",
      autocapitalize: "autoCapitalize",
      autocomplete: "autoComplete",
      autocorrect: "autoCorrect",
      autofocus: "autoFocus",
      autoplay: "autoPlay",
      autosave: "autoSave",
      capture: "capture",
      cellpadding: "cellPadding",
      cellspacing: "cellSpacing",
      challenge: "challenge",
      charset: "charSet",
      checked: "checked",
      children: "children",
      cite: "cite",
      class: "className",
      classid: "classID",
      classname: "className",
      cols: "cols",
      colspan: "colSpan",
      content: "content",
      contenteditable: "contentEditable",
      contextmenu: "contextMenu",
      controls: "controls",
      controlslist: "controlsList",
      coords: "coords",
      crossorigin: "crossOrigin",
      dangerouslysetinnerhtml: "dangerouslySetInnerHTML",
      data: "data",
      datetime: "dateTime",
      default: "default",
      defaultchecked: "defaultChecked",
      defaultvalue: "defaultValue",
      defer: "defer",
      dir: "dir",
      disabled: "disabled",
      disablepictureinpicture: "disablePictureInPicture",
      disableremoteplayback: "disableRemotePlayback",
      download: "download",
      draggable: "draggable",
      enctype: "encType",
      enterkeyhint: "enterKeyHint",
      fetchpriority: "fetchPriority",
      for: "htmlFor",
      form: "form",
      formmethod: "formMethod",
      formaction: "formAction",
      formenctype: "formEncType",
      formnovalidate: "formNoValidate",
      formtarget: "formTarget",
      frameborder: "frameBorder",
      headers: "headers",
      height: "height",
      hidden: "hidden",
      high: "high",
      href: "href",
      hreflang: "hrefLang",
      htmlfor: "htmlFor",
      httpequiv: "httpEquiv",
      "http-equiv": "httpEquiv",
      icon: "icon",
      id: "id",
      imagesizes: "imageSizes",
      imagesrcset: "imageSrcSet",
      inert: "inert",
      innerhtml: "innerHTML",
      inputmode: "inputMode",
      integrity: "integrity",
      is: "is",
      itemid: "itemID",
      itemprop: "itemProp",
      itemref: "itemRef",
      itemscope: "itemScope",
      itemtype: "itemType",
      keyparams: "keyParams",
      keytype: "keyType",
      kind: "kind",
      label: "label",
      lang: "lang",
      list: "list",
      loop: "loop",
      low: "low",
      manifest: "manifest",
      marginwidth: "marginWidth",
      marginheight: "marginHeight",
      max: "max",
      maxlength: "maxLength",
      media: "media",
      mediagroup: "mediaGroup",
      method: "method",
      min: "min",
      minlength: "minLength",
      multiple: "multiple",
      muted: "muted",
      name: "name",
      nomodule: "noModule",
      nonce: "nonce",
      novalidate: "noValidate",
      open: "open",
      optimum: "optimum",
      pattern: "pattern",
      placeholder: "placeholder",
      playsinline: "playsInline",
      poster: "poster",
      preload: "preload",
      profile: "profile",
      radiogroup: "radioGroup",
      readonly: "readOnly",
      referrerpolicy: "referrerPolicy",
      rel: "rel",
      required: "required",
      reversed: "reversed",
      role: "role",
      rows: "rows",
      rowspan: "rowSpan",
      sandbox: "sandbox",
      scope: "scope",
      scoped: "scoped",
      scrolling: "scrolling",
      seamless: "seamless",
      selected: "selected",
      shape: "shape",
      size: "size",
      sizes: "sizes",
      span: "span",
      spellcheck: "spellCheck",
      src: "src",
      srcdoc: "srcDoc",
      srclang: "srcLang",
      srcset: "srcSet",
      start: "start",
      step: "step",
      style: "style",
      summary: "summary",
      tabindex: "tabIndex",
      target: "target",
      title: "title",
      type: "type",
      usemap: "useMap",
      value: "value",
      width: "width",
      wmode: "wmode",
      wrap: "wrap",
      about: "about",
      accentheight: "accentHeight",
      "accent-height": "accentHeight",
      accumulate: "accumulate",
      additive: "additive",
      alignmentbaseline: "alignmentBaseline",
      "alignment-baseline": "alignmentBaseline",
      allowreorder: "allowReorder",
      alphabetic: "alphabetic",
      amplitude: "amplitude",
      arabicform: "arabicForm",
      "arabic-form": "arabicForm",
      ascent: "ascent",
      attributename: "attributeName",
      attributetype: "attributeType",
      autoreverse: "autoReverse",
      azimuth: "azimuth",
      basefrequency: "baseFrequency",
      baselineshift: "baselineShift",
      "baseline-shift": "baselineShift",
      baseprofile: "baseProfile",
      bbox: "bbox",
      begin: "begin",
      bias: "bias",
      by: "by",
      calcmode: "calcMode",
      capheight: "capHeight",
      "cap-height": "capHeight",
      clip: "clip",
      clippath: "clipPath",
      "clip-path": "clipPath",
      clippathunits: "clipPathUnits",
      cliprule: "clipRule",
      "clip-rule": "clipRule",
      color: "color",
      colorinterpolation: "colorInterpolation",
      "color-interpolation": "colorInterpolation",
      colorinterpolationfilters: "colorInterpolationFilters",
      "color-interpolation-filters": "colorInterpolationFilters",
      colorprofile: "colorProfile",
      "color-profile": "colorProfile",
      colorrendering: "colorRendering",
      "color-rendering": "colorRendering",
      contentscripttype: "contentScriptType",
      contentstyletype: "contentStyleType",
      cursor: "cursor",
      cx: "cx",
      cy: "cy",
      d: "d",
      datatype: "datatype",
      decelerate: "decelerate",
      descent: "descent",
      diffuseconstant: "diffuseConstant",
      direction: "direction",
      display: "display",
      divisor: "divisor",
      dominantbaseline: "dominantBaseline",
      "dominant-baseline": "dominantBaseline",
      dur: "dur",
      dx: "dx",
      dy: "dy",
      edgemode: "edgeMode",
      elevation: "elevation",
      enablebackground: "enableBackground",
      "enable-background": "enableBackground",
      end: "end",
      exponent: "exponent",
      externalresourcesrequired: "externalResourcesRequired",
      fill: "fill",
      fillopacity: "fillOpacity",
      "fill-opacity": "fillOpacity",
      fillrule: "fillRule",
      "fill-rule": "fillRule",
      filter: "filter",
      filterres: "filterRes",
      filterunits: "filterUnits",
      floodopacity: "floodOpacity",
      "flood-opacity": "floodOpacity",
      floodcolor: "floodColor",
      "flood-color": "floodColor",
      focusable: "focusable",
      fontfamily: "fontFamily",
      "font-family": "fontFamily",
      fontsize: "fontSize",
      "font-size": "fontSize",
      fontsizeadjust: "fontSizeAdjust",
      "font-size-adjust": "fontSizeAdjust",
      fontstretch: "fontStretch",
      "font-stretch": "fontStretch",
      fontstyle: "fontStyle",
      "font-style": "fontStyle",
      fontvariant: "fontVariant",
      "font-variant": "fontVariant",
      fontweight: "fontWeight",
      "font-weight": "fontWeight",
      format: "format",
      from: "from",
      fx: "fx",
      fy: "fy",
      g1: "g1",
      g2: "g2",
      glyphname: "glyphName",
      "glyph-name": "glyphName",
      glyphorientationhorizontal: "glyphOrientationHorizontal",
      "glyph-orientation-horizontal": "glyphOrientationHorizontal",
      glyphorientationvertical: "glyphOrientationVertical",
      "glyph-orientation-vertical": "glyphOrientationVertical",
      glyphref: "glyphRef",
      gradienttransform: "gradientTransform",
      gradientunits: "gradientUnits",
      hanging: "hanging",
      horizadvx: "horizAdvX",
      "horiz-adv-x": "horizAdvX",
      horizoriginx: "horizOriginX",
      "horiz-origin-x": "horizOriginX",
      ideographic: "ideographic",
      imagerendering: "imageRendering",
      "image-rendering": "imageRendering",
      in2: "in2",
      in: "in",
      inlist: "inlist",
      intercept: "intercept",
      k1: "k1",
      k2: "k2",
      k3: "k3",
      k4: "k4",
      k: "k",
      kernelmatrix: "kernelMatrix",
      kernelunitlength: "kernelUnitLength",
      kerning: "kerning",
      keypoints: "keyPoints",
      keysplines: "keySplines",
      keytimes: "keyTimes",
      lengthadjust: "lengthAdjust",
      letterspacing: "letterSpacing",
      "letter-spacing": "letterSpacing",
      lightingcolor: "lightingColor",
      "lighting-color": "lightingColor",
      limitingconeangle: "limitingConeAngle",
      local: "local",
      markerend: "markerEnd",
      "marker-end": "markerEnd",
      markerheight: "markerHeight",
      markermid: "markerMid",
      "marker-mid": "markerMid",
      markerstart: "markerStart",
      "marker-start": "markerStart",
      markerunits: "markerUnits",
      markerwidth: "markerWidth",
      mask: "mask",
      maskcontentunits: "maskContentUnits",
      maskunits: "maskUnits",
      mathematical: "mathematical",
      mode: "mode",
      numoctaves: "numOctaves",
      offset: "offset",
      opacity: "opacity",
      operator: "operator",
      order: "order",
      orient: "orient",
      orientation: "orientation",
      origin: "origin",
      overflow: "overflow",
      overlineposition: "overlinePosition",
      "overline-position": "overlinePosition",
      overlinethickness: "overlineThickness",
      "overline-thickness": "overlineThickness",
      paintorder: "paintOrder",
      "paint-order": "paintOrder",
      panose1: "panose1",
      "panose-1": "panose1",
      pathlength: "pathLength",
      patterncontentunits: "patternContentUnits",
      patterntransform: "patternTransform",
      patternunits: "patternUnits",
      pointerevents: "pointerEvents",
      "pointer-events": "pointerEvents",
      points: "points",
      pointsatx: "pointsAtX",
      pointsaty: "pointsAtY",
      pointsatz: "pointsAtZ",
      popover: "popover",
      popovertarget: "popoverTarget",
      popovertargetaction: "popoverTargetAction",
      prefix: "prefix",
      preservealpha: "preserveAlpha",
      preserveaspectratio: "preserveAspectRatio",
      primitiveunits: "primitiveUnits",
      property: "property",
      r: "r",
      radius: "radius",
      refx: "refX",
      refy: "refY",
      renderingintent: "renderingIntent",
      "rendering-intent": "renderingIntent",
      repeatcount: "repeatCount",
      repeatdur: "repeatDur",
      requiredextensions: "requiredExtensions",
      requiredfeatures: "requiredFeatures",
      resource: "resource",
      restart: "restart",
      result: "result",
      results: "results",
      rotate: "rotate",
      rx: "rx",
      ry: "ry",
      scale: "scale",
      security: "security",
      seed: "seed",
      shaperendering: "shapeRendering",
      "shape-rendering": "shapeRendering",
      slope: "slope",
      spacing: "spacing",
      specularconstant: "specularConstant",
      specularexponent: "specularExponent",
      speed: "speed",
      spreadmethod: "spreadMethod",
      startoffset: "startOffset",
      stddeviation: "stdDeviation",
      stemh: "stemh",
      stemv: "stemv",
      stitchtiles: "stitchTiles",
      stopcolor: "stopColor",
      "stop-color": "stopColor",
      stopopacity: "stopOpacity",
      "stop-opacity": "stopOpacity",
      strikethroughposition: "strikethroughPosition",
      "strikethrough-position": "strikethroughPosition",
      strikethroughthickness: "strikethroughThickness",
      "strikethrough-thickness": "strikethroughThickness",
      string: "string",
      stroke: "stroke",
      strokedasharray: "strokeDasharray",
      "stroke-dasharray": "strokeDasharray",
      strokedashoffset: "strokeDashoffset",
      "stroke-dashoffset": "strokeDashoffset",
      strokelinecap: "strokeLinecap",
      "stroke-linecap": "strokeLinecap",
      strokelinejoin: "strokeLinejoin",
      "stroke-linejoin": "strokeLinejoin",
      strokemiterlimit: "strokeMiterlimit",
      "stroke-miterlimit": "strokeMiterlimit",
      strokewidth: "strokeWidth",
      "stroke-width": "strokeWidth",
      strokeopacity: "strokeOpacity",
      "stroke-opacity": "strokeOpacity",
      suppresscontenteditablewarning: "suppressContentEditableWarning",
      suppresshydrationwarning: "suppressHydrationWarning",
      surfacescale: "surfaceScale",
      systemlanguage: "systemLanguage",
      tablevalues: "tableValues",
      targetx: "targetX",
      targety: "targetY",
      textanchor: "textAnchor",
      "text-anchor": "textAnchor",
      textdecoration: "textDecoration",
      "text-decoration": "textDecoration",
      textlength: "textLength",
      textrendering: "textRendering",
      "text-rendering": "textRendering",
      to: "to",
      transform: "transform",
      transformorigin: "transformOrigin",
      "transform-origin": "transformOrigin",
      typeof: "typeof",
      u1: "u1",
      u2: "u2",
      underlineposition: "underlinePosition",
      "underline-position": "underlinePosition",
      underlinethickness: "underlineThickness",
      "underline-thickness": "underlineThickness",
      unicode: "unicode",
      unicodebidi: "unicodeBidi",
      "unicode-bidi": "unicodeBidi",
      unicoderange: "unicodeRange",
      "unicode-range": "unicodeRange",
      unitsperem: "unitsPerEm",
      "units-per-em": "unitsPerEm",
      unselectable: "unselectable",
      valphabetic: "vAlphabetic",
      "v-alphabetic": "vAlphabetic",
      values: "values",
      vectoreffect: "vectorEffect",
      "vector-effect": "vectorEffect",
      version: "version",
      vertadvy: "vertAdvY",
      "vert-adv-y": "vertAdvY",
      vertoriginx: "vertOriginX",
      "vert-origin-x": "vertOriginX",
      vertoriginy: "vertOriginY",
      "vert-origin-y": "vertOriginY",
      vhanging: "vHanging",
      "v-hanging": "vHanging",
      videographic: "vIdeographic",
      "v-ideographic": "vIdeographic",
      viewbox: "viewBox",
      viewtarget: "viewTarget",
      visibility: "visibility",
      vmathematical: "vMathematical",
      "v-mathematical": "vMathematical",
      vocab: "vocab",
      widths: "widths",
      wordspacing: "wordSpacing",
      "word-spacing": "wordSpacing",
      writingmode: "writingMode",
      "writing-mode": "writingMode",
      x1: "x1",
      x2: "x2",
      x: "x",
      xchannelselector: "xChannelSelector",
      xheight: "xHeight",
      "x-height": "xHeight",
      xlinkactuate: "xlinkActuate",
      "xlink:actuate": "xlinkActuate",
      xlinkarcrole: "xlinkArcrole",
      "xlink:arcrole": "xlinkArcrole",
      xlinkhref: "xlinkHref",
      "xlink:href": "xlinkHref",
      xlinkrole: "xlinkRole",
      "xlink:role": "xlinkRole",
      xlinkshow: "xlinkShow",
      "xlink:show": "xlinkShow",
      xlinktitle: "xlinkTitle",
      "xlink:title": "xlinkTitle",
      xlinktype: "xlinkType",
      "xlink:type": "xlinkType",
      xmlbase: "xmlBase",
      "xml:base": "xmlBase",
      xmllang: "xmlLang",
      "xml:lang": "xmlLang",
      xmlns: "xmlns",
      "xml:space": "xmlSpace",
      xmlnsxlink: "xmlnsXlink",
      "xmlns:xlink": "xmlnsXlink",
      xmlspace: "xmlSpace",
      y1: "y1",
      y2: "y2",
      y: "y",
      ychannelselector: "yChannelSelector",
      z: "z",
      zoomandpan: "zoomAndPan"
    }, Bm = {
      "aria-current": 0,
      "aria-description": 0,
      "aria-details": 0,
      "aria-disabled": 0,
      "aria-hidden": 0,
      "aria-invalid": 0,
      "aria-keyshortcuts": 0,
      "aria-label": 0,
      "aria-roledescription": 0,
      "aria-autocomplete": 0,
      "aria-checked": 0,
      "aria-expanded": 0,
      "aria-haspopup": 0,
      "aria-level": 0,
      "aria-modal": 0,
      "aria-multiline": 0,
      "aria-multiselectable": 0,
      "aria-orientation": 0,
      "aria-placeholder": 0,
      "aria-pressed": 0,
      "aria-readonly": 0,
      "aria-required": 0,
      "aria-selected": 0,
      "aria-sort": 0,
      "aria-valuemax": 0,
      "aria-valuemin": 0,
      "aria-valuenow": 0,
      "aria-valuetext": 0,
      "aria-atomic": 0,
      "aria-busy": 0,
      "aria-live": 0,
      "aria-relevant": 0,
      "aria-dropeffect": 0,
      "aria-grabbed": 0,
      "aria-activedescendant": 0,
      "aria-colcount": 0,
      "aria-colindex": 0,
      "aria-colspan": 0,
      "aria-controls": 0,
      "aria-describedby": 0,
      "aria-errormessage": 0,
      "aria-flowto": 0,
      "aria-labelledby": 0,
      "aria-owns": 0,
      "aria-posinset": 0,
      "aria-rowcount": 0,
      "aria-rowindex": 0,
      "aria-rowspan": 0,
      "aria-setsize": 0
    }, iu = {}, Ym = RegExp(
      "^(aria)-[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), eh = RegExp(
      "^(aria)[A-Z][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), Gm = !1, ea = {}, Gs = /^on./, l = /^on[^A-Z]/, n = RegExp(
      "^(aria)-[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), u = RegExp(
      "^(aria)[A-Z][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), c = /^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i, s = null, r = null, y = null, p = !1, S = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), H = !1;
    if (S)
      try {
        var K = {};
        Object.defineProperty(K, "passive", {
          get: function() {
            H = !0;
          }
        }), window.addEventListener("test", K, K), window.removeEventListener("test", K, K);
      } catch {
        H = !1;
      }
    var $ = null, q = null, Y = null, Ae = {
      eventPhase: 0,
      bubbles: 0,
      cancelable: 0,
      timeStamp: function(e) {
        return e.timeStamp || Date.now();
      },
      defaultPrevented: 0,
      isTrusted: 0
    }, Re = _l(Ae), yt = Je({}, Ae, { view: 0, detail: 0 }), M = _l(yt), R, C, k, de = Je({}, yt, {
      screenX: 0,
      screenY: 0,
      clientX: 0,
      clientY: 0,
      pageX: 0,
      pageY: 0,
      ctrlKey: 0,
      shiftKey: 0,
      altKey: 0,
      metaKey: 0,
      getModifierState: br,
      button: 0,
      buttons: 0,
      relatedTarget: function(e) {
        return e.relatedTarget === void 0 ? e.fromElement === e.srcElement ? e.toElement : e.fromElement : e.relatedTarget;
      },
      movementX: function(e) {
        return "movementX" in e ? e.movementX : (e !== k && (k && e.type === "mousemove" ? (R = e.screenX - k.screenX, C = e.screenY - k.screenY) : C = R = 0, k = e), R);
      },
      movementY: function(e) {
        return "movementY" in e ? e.movementY : C;
      }
    }), We = _l(de), Te = Je({}, de, { dataTransfer: 0 }), _e = _l(Te), El = Je({}, yt, { relatedTarget: 0 }), st = _l(El), Bi = Je({}, Ae, {
      animationName: 0,
      elapsedTime: 0,
      pseudoElement: 0
    }), Lg = _l(Bi), oS = Je({}, Ae, {
      clipboardData: function(e) {
        return "clipboardData" in e ? e.clipboardData : window.clipboardData;
      }
    }), fS = _l(oS), sS = Je({}, Ae, { data: 0 }), X0 = _l(
      sS
    ), rS = X0, dS = {
      Esc: "Escape",
      Spacebar: " ",
      Left: "ArrowLeft",
      Up: "ArrowUp",
      Right: "ArrowRight",
      Down: "ArrowDown",
      Del: "Delete",
      Win: "OS",
      Menu: "ContextMenu",
      Apps: "ContextMenu",
      Scroll: "ScrollLock",
      MozPrintableKey: "Unidentified"
    }, hS = {
      8: "Backspace",
      9: "Tab",
      12: "Clear",
      13: "Enter",
      16: "Shift",
      17: "Control",
      18: "Alt",
      19: "Pause",
      20: "CapsLock",
      27: "Escape",
      32: " ",
      33: "PageUp",
      34: "PageDown",
      35: "End",
      36: "Home",
      37: "ArrowLeft",
      38: "ArrowUp",
      39: "ArrowRight",
      40: "ArrowDown",
      45: "Insert",
      46: "Delete",
      112: "F1",
      113: "F2",
      114: "F3",
      115: "F4",
      116: "F5",
      117: "F6",
      118: "F7",
      119: "F8",
      120: "F9",
      121: "F10",
      122: "F11",
      123: "F12",
      144: "NumLock",
      145: "ScrollLock",
      224: "Meta"
    }, yS = {
      Alt: "altKey",
      Control: "ctrlKey",
      Meta: "metaKey",
      Shift: "shiftKey"
    }, mS = Je({}, yt, {
      key: function(e) {
        if (e.key) {
          var t = dS[e.key] || e.key;
          if (t !== "Unidentified") return t;
        }
        return e.type === "keypress" ? (e = fo(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? hS[e.keyCode] || "Unidentified" : "";
      },
      code: 0,
      location: 0,
      ctrlKey: 0,
      shiftKey: 0,
      altKey: 0,
      metaKey: 0,
      repeat: 0,
      locale: 0,
      getModifierState: br,
      charCode: function(e) {
        return e.type === "keypress" ? fo(e) : 0;
      },
      keyCode: function(e) {
        return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
      },
      which: function(e) {
        return e.type === "keypress" ? fo(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
      }
    }), pS = _l(mS), vS = Je({}, de, {
      pointerId: 0,
      width: 0,
      height: 0,
      pressure: 0,
      tangentialPressure: 0,
      tiltX: 0,
      tiltY: 0,
      twist: 0,
      pointerType: 0,
      isPrimary: 0
    }), Q0 = _l(vS), gS = Je({}, yt, {
      touches: 0,
      targetTouches: 0,
      changedTouches: 0,
      altKey: 0,
      metaKey: 0,
      ctrlKey: 0,
      shiftKey: 0,
      getModifierState: br
    }), bS = _l(gS), SS = Je({}, Ae, {
      propertyName: 0,
      elapsedTime: 0,
      pseudoElement: 0
    }), TS = _l(SS), ES = Je({}, de, {
      deltaX: function(e) {
        return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
      },
      deltaY: function(e) {
        return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
      },
      deltaZ: 0,
      deltaMode: 0
    }), AS = _l(ES), RS = Je({}, Ae, {
      newState: 0,
      oldState: 0
    }), OS = _l(RS), DS = [9, 13, 27, 32], Z0 = 229, Vg = S && "CompositionEvent" in window, Lm = null;
    S && "documentMode" in document && (Lm = document.documentMode);
    var zS = S && "TextEvent" in window && !Lm, K0 = S && (!Vg || Lm && 8 < Lm && 11 >= Lm), J0 = 32, k0 = String.fromCharCode(J0), $0 = !1, th = !1, MS = {
      color: !0,
      date: !0,
      datetime: !0,
      "datetime-local": !0,
      email: !0,
      month: !0,
      number: !0,
      password: !0,
      range: !0,
      search: !0,
      tel: !0,
      text: !0,
      time: !0,
      url: !0,
      week: !0
    }, Vm = null, Xm = null, W0 = !1;
    S && (W0 = Lh("input") && (!document.documentMode || 9 < document.documentMode));
    var Ha = typeof Object.is == "function" ? Object.is : zg, _S = S && "documentMode" in document && 11 >= document.documentMode, lh = null, Xg = null, Qm = null, Qg = !1, ah = {
      animationend: Au("Animation", "AnimationEnd"),
      animationiteration: Au("Animation", "AnimationIteration"),
      animationstart: Au("Animation", "AnimationStart"),
      transitionrun: Au("Transition", "TransitionRun"),
      transitionstart: Au("Transition", "TransitionStart"),
      transitioncancel: Au("Transition", "TransitionCancel"),
      transitionend: Au("Transition", "TransitionEnd")
    }, Zg = {}, F0 = {};
    S && (F0 = document.createElement("div").style, "AnimationEvent" in window || (delete ah.animationend.animation, delete ah.animationiteration.animation, delete ah.animationstart.animation), "TransitionEvent" in window || delete ah.transitionend.transition);
    var I0 = lc("animationend"), P0 = lc("animationiteration"), e1 = lc("animationstart"), US = lc("transitionrun"), CS = lc("transitionstart"), xS = lc("transitioncancel"), t1 = lc("transitionend"), l1 = /* @__PURE__ */ new Map(), Kg = "abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(
      " "
    );
    Kg.push("scrollEnd");
    var Jg = /* @__PURE__ */ new WeakMap(), Nv = 1, Yc = 2, cu = [], nh = 0, kg = 0, nf = {};
    Object.freeze(nf);
    var ou = null, uh = null, Bt = 0, HS = 1, ta = 2, Sa = 8, Ku = 16, a1 = 64, n1 = !1;
    try {
      var u1 = Object.preventExtensions({});
    } catch {
      n1 = !0;
    }
    var ih = [], ch = 0, wv = null, qv = 0, fu = [], su = 0, Ls = null, Gc = 1, Lc = "", Na = null, nl = null, mt = !1, Vc = !1, ru = null, Vs = null, Yi = !1, $g = Error(
      "Hydration Mismatch Exception: This is not a real error, and should not leak into userspace. If you're seeing this, it's likely a bug in React."
    ), i1 = 0;
    if (typeof performance == "object" && typeof performance.now == "function")
      var NS = performance, c1 = function() {
        return NS.now();
      };
    else {
      var wS = Date;
      c1 = function() {
        return wS.now();
      };
    }
    var Wg = Rt(null), Fg = Rt(null), o1 = {}, jv = null, oh = null, fh = !1, qS = typeof AbortController < "u" ? AbortController : function() {
      var e = [], t = this.signal = {
        aborted: !1,
        addEventListener: function(a, i) {
          e.push(i);
        }
      };
      this.abort = function() {
        t.aborted = !0, e.forEach(function(a) {
          return a();
        });
      };
    }, jS = Wt.unstable_scheduleCallback, BS = Wt.unstable_NormalPriority, jl = {
      $$typeof: Ia,
      Consumer: null,
      Provider: null,
      _currentValue: null,
      _currentValue2: null,
      _threadCount: 0,
      _currentRenderer: null,
      _currentRenderer2: null
    }, sh = Wt.unstable_now, f1 = -0, Bv = -0, tn = -1.1, Xs = -0, Yv = !1, Gv = !1, Zm = null, Ig = 0, Qs = 0, rh = null, s1 = L.S;
    L.S = function(e, t) {
      typeof t == "object" && t !== null && typeof t.then == "function" && Qp(e, t), s1 !== null && s1(e, t);
    };
    var Zs = Rt(null), Ju = {
      recordUnsafeLifecycleWarnings: function() {
      },
      flushPendingUnsafeLifecycleWarnings: function() {
      },
      recordLegacyContextWarning: function() {
      },
      flushLegacyContextWarning: function() {
      },
      discardPendingWarnings: function() {
      }
    }, Km = [], Jm = [], km = [], $m = [], Wm = [], Fm = [], Ks = /* @__PURE__ */ new Set();
    Ju.recordUnsafeLifecycleWarnings = function(e, t) {
      Ks.has(e.type) || (typeof t.componentWillMount == "function" && t.componentWillMount.__suppressDeprecationWarning !== !0 && Km.push(e), e.mode & Sa && typeof t.UNSAFE_componentWillMount == "function" && Jm.push(e), typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps.__suppressDeprecationWarning !== !0 && km.push(e), e.mode & Sa && typeof t.UNSAFE_componentWillReceiveProps == "function" && $m.push(e), typeof t.componentWillUpdate == "function" && t.componentWillUpdate.__suppressDeprecationWarning !== !0 && Wm.push(e), e.mode & Sa && typeof t.UNSAFE_componentWillUpdate == "function" && Fm.push(e));
    }, Ju.flushPendingUnsafeLifecycleWarnings = function() {
      var e = /* @__PURE__ */ new Set();
      0 < Km.length && (Km.forEach(function(h) {
        e.add(
          re(h) || "Component"
        ), Ks.add(h.type);
      }), Km = []);
      var t = /* @__PURE__ */ new Set();
      0 < Jm.length && (Jm.forEach(function(h) {
        t.add(
          re(h) || "Component"
        ), Ks.add(h.type);
      }), Jm = []);
      var a = /* @__PURE__ */ new Set();
      0 < km.length && (km.forEach(function(h) {
        a.add(
          re(h) || "Component"
        ), Ks.add(h.type);
      }), km = []);
      var i = /* @__PURE__ */ new Set();
      0 < $m.length && ($m.forEach(
        function(h) {
          i.add(
            re(h) || "Component"
          ), Ks.add(h.type);
        }
      ), $m = []);
      var o = /* @__PURE__ */ new Set();
      0 < Wm.length && (Wm.forEach(function(h) {
        o.add(
          re(h) || "Component"
        ), Ks.add(h.type);
      }), Wm = []);
      var f = /* @__PURE__ */ new Set();
      if (0 < Fm.length && (Fm.forEach(function(h) {
        f.add(
          re(h) || "Component"
        ), Ks.add(h.type);
      }), Fm = []), 0 < t.size) {
        var d = G(
          t
        );
        console.error(
          `Using UNSAFE_componentWillMount in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move code with side effects to componentDidMount, and set initial state in the constructor.

Please update the following components: %s`,
          d
        );
      }
      0 < i.size && (d = G(
        i
      ), console.error(
        `Using UNSAFE_componentWillReceiveProps in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* If you're updating state whenever props change, refactor your code to use memoization techniques or move it to static getDerivedStateFromProps. Learn more at: https://react.dev/link/derived-state

Please update the following components: %s`,
        d
      )), 0 < f.size && (d = G(
        f
      ), console.error(
        `Using UNSAFE_componentWillUpdate in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.

Please update the following components: %s`,
        d
      )), 0 < e.size && (d = G(e), console.warn(
        `componentWillMount has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move code with side effects to componentDidMount, and set initial state in the constructor.
* Rename componentWillMount to UNSAFE_componentWillMount to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      )), 0 < a.size && (d = G(
        a
      ), console.warn(
        `componentWillReceiveProps has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* If you're updating state whenever props change, refactor your code to use memoization techniques or move it to static getDerivedStateFromProps. Learn more at: https://react.dev/link/derived-state
* Rename componentWillReceiveProps to UNSAFE_componentWillReceiveProps to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      )), 0 < o.size && (d = G(o), console.warn(
        `componentWillUpdate has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* Rename componentWillUpdate to UNSAFE_componentWillUpdate to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      ));
    };
    var Lv = /* @__PURE__ */ new Map(), r1 = /* @__PURE__ */ new Set();
    Ju.recordLegacyContextWarning = function(e, t) {
      for (var a = null, i = e; i !== null; )
        i.mode & Sa && (a = i), i = i.return;
      a === null ? console.error(
        "Expected to find a StrictMode component in a strict mode tree. This error is likely caused by a bug in React. Please file an issue."
      ) : !r1.has(e.type) && (i = Lv.get(a), e.type.contextTypes != null || e.type.childContextTypes != null || t !== null && typeof t.getChildContext == "function") && (i === void 0 && (i = [], Lv.set(a, i)), i.push(e));
    }, Ju.flushLegacyContextWarning = function() {
      Lv.forEach(function(e) {
        if (e.length !== 0) {
          var t = e[0], a = /* @__PURE__ */ new Set();
          e.forEach(function(o) {
            a.add(re(o) || "Component"), r1.add(o.type);
          });
          var i = G(a);
          ye(t, function() {
            console.error(
              `Legacy context API has been detected within a strict-mode tree.

The old API will be supported in all 16.x releases, but applications using it should migrate to the new version.

Please update the following components: %s

Learn more about this warning here: https://react.dev/link/legacy-context`,
              i
            );
          });
        }
      });
    }, Ju.discardPendingWarnings = function() {
      Km = [], Jm = [], km = [], $m = [], Wm = [], Fm = [], Lv = /* @__PURE__ */ new Map();
    };
    var Im = Error(
      "Suspense Exception: This is not a real error! It's an implementation detail of `use` to interrupt the current render. You must either rethrow it immediately, or move the `use` call outside of the `try/catch` block. Capturing without rethrowing will lead to unexpected behavior.\n\nTo handle async errors, wrap your component in an error boundary, or call the promise's `.catch` method and pass the result to `use`."
    ), d1 = Error(
      "Suspense Exception: This is not a real error, and should not leak into userspace. If you're seeing this, it's likely a bug in React."
    ), Vv = Error(
      "Suspense Exception: This is not a real error! It's an implementation detail of `useActionState` to interrupt the current render. You must either rethrow it immediately, or move the `useActionState` call outside of the `try/catch` block. Capturing without rethrowing will lead to unexpected behavior.\n\nTo handle async errors, wrap your component in an error boundary."
    ), Pg = {
      then: function() {
        console.error(
          'Internal React error: A listener was unexpectedly attached to a "noop" thenable. This is a bug in React. Please file an issue.'
        );
      }
    }, Pm = null, Xv = !1, du = 0, hu = 1, wa = 2, la = 4, Bl = 8, h1 = 0, y1 = 1, m1 = 2, e0 = 3, uf = !1, p1 = !1, t0 = null, l0 = !1, dh = Rt(null), Qv = Rt(0), hh, v1 = /* @__PURE__ */ new Set(), g1 = /* @__PURE__ */ new Set(), a0 = /* @__PURE__ */ new Set(), b1 = /* @__PURE__ */ new Set(), cf = 0, qe = null, Ut = null, Al = null, Zv = !1, yh = !1, Js = !1, Kv = 0, ep = 0, Xc = null, YS = 0, GS = 25, V = null, yu = null, Qc = -1, tp = !1, Jv = {
      readContext: Ct,
      use: jn,
      useCallback: Lt,
      useContext: Lt,
      useEffect: Lt,
      useImperativeHandle: Lt,
      useLayoutEffect: Lt,
      useInsertionEffect: Lt,
      useMemo: Lt,
      useReducer: Lt,
      useRef: Lt,
      useState: Lt,
      useDebugValue: Lt,
      useDeferredValue: Lt,
      useTransition: Lt,
      useSyncExternalStore: Lt,
      useId: Lt,
      useHostTransitionStatus: Lt,
      useFormState: Lt,
      useActionState: Lt,
      useOptimistic: Lt,
      useMemoCache: Lt,
      useCacheRefresh: Lt
    }, n0 = null, S1 = null, u0 = null, T1 = null, Gi = null, ku = null, kv = null;
    n0 = {
      readContext: function(e) {
        return Ct(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return V = "useCallback", $e(), Va(t), Jf(e, t);
      },
      useContext: function(e) {
        return V = "useContext", $e(), Ct(e);
      },
      useEffect: function(e, t) {
        return V = "useEffect", $e(), Va(t), Hr(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return V = "useImperativeHandle", $e(), Va(a), wr(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        V = "useInsertionEffect", $e(), Va(t), Ka(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return V = "useLayoutEffect", $e(), Va(t), Nr(e, t);
      },
      useMemo: function(e, t) {
        V = "useMemo", $e(), Va(t);
        var a = L.H;
        L.H = Gi;
        try {
          return qr(e, t);
        } finally {
          L.H = a;
        }
      },
      useReducer: function(e, t, a) {
        V = "useReducer", $e();
        var i = L.H;
        L.H = Gi;
        try {
          return rt(e, t, a);
        } finally {
          L.H = i;
        }
      },
      useRef: function(e) {
        return V = "useRef", $e(), Kf(e);
      },
      useState: function(e) {
        V = "useState", $e();
        var t = L.H;
        L.H = Gi;
        try {
          return Mu(e);
        } finally {
          L.H = t;
        }
      },
      useDebugValue: function() {
        V = "useDebugValue", $e();
      },
      useDeferredValue: function(e, t) {
        return V = "useDeferredValue", $e(), jr(e, t);
      },
      useTransition: function() {
        return V = "useTransition", $e(), Ln();
      },
      useSyncExternalStore: function(e, t, a) {
        return V = "useSyncExternalStore", $e(), zu(
          e,
          t,
          a
        );
      },
      useId: function() {
        return V = "useId", $e(), Vn();
      },
      useFormState: function(e, t) {
        return V = "useFormState", $e(), po(), Eo(e, t);
      },
      useActionState: function(e, t) {
        return V = "useActionState", $e(), Eo(e, t);
      },
      useOptimistic: function(e) {
        return V = "useOptimistic", $e(), pn(e);
      },
      useHostTransitionStatus: sa,
      useMemoCache: Pt,
      useCacheRefresh: function() {
        return V = "useCacheRefresh", $e(), yc();
      }
    }, S1 = {
      readContext: function(e) {
        return Ct(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return V = "useCallback", ee(), Jf(e, t);
      },
      useContext: function(e) {
        return V = "useContext", ee(), Ct(e);
      },
      useEffect: function(e, t) {
        return V = "useEffect", ee(), Hr(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return V = "useImperativeHandle", ee(), wr(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        V = "useInsertionEffect", ee(), Ka(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return V = "useLayoutEffect", ee(), Nr(e, t);
      },
      useMemo: function(e, t) {
        V = "useMemo", ee();
        var a = L.H;
        L.H = Gi;
        try {
          return qr(e, t);
        } finally {
          L.H = a;
        }
      },
      useReducer: function(e, t, a) {
        V = "useReducer", ee();
        var i = L.H;
        L.H = Gi;
        try {
          return rt(e, t, a);
        } finally {
          L.H = i;
        }
      },
      useRef: function(e) {
        return V = "useRef", ee(), Kf(e);
      },
      useState: function(e) {
        V = "useState", ee();
        var t = L.H;
        L.H = Gi;
        try {
          return Mu(e);
        } finally {
          L.H = t;
        }
      },
      useDebugValue: function() {
        V = "useDebugValue", ee();
      },
      useDeferredValue: function(e, t) {
        return V = "useDeferredValue", ee(), jr(e, t);
      },
      useTransition: function() {
        return V = "useTransition", ee(), Ln();
      },
      useSyncExternalStore: function(e, t, a) {
        return V = "useSyncExternalStore", ee(), zu(
          e,
          t,
          a
        );
      },
      useId: function() {
        return V = "useId", ee(), Vn();
      },
      useActionState: function(e, t) {
        return V = "useActionState", ee(), Eo(e, t);
      },
      useFormState: function(e, t) {
        return V = "useFormState", ee(), po(), Eo(e, t);
      },
      useOptimistic: function(e) {
        return V = "useOptimistic", ee(), pn(e);
      },
      useHostTransitionStatus: sa,
      useMemoCache: Pt,
      useCacheRefresh: function() {
        return V = "useCacheRefresh", ee(), yc();
      }
    }, u0 = {
      readContext: function(e) {
        return Ct(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return V = "useCallback", ee(), dc(e, t);
      },
      useContext: function(e) {
        return V = "useContext", ee(), Ct(e);
      },
      useEffect: function(e, t) {
        V = "useEffect", ee(), sl(2048, Bl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return V = "useImperativeHandle", ee(), Gn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return V = "useInsertionEffect", ee(), sl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return V = "useLayoutEffect", ee(), sl(4, la, e, t);
      },
      useMemo: function(e, t) {
        V = "useMemo", ee();
        var a = L.H;
        L.H = ku;
        try {
          return vi(e, t);
        } finally {
          L.H = a;
        }
      },
      useReducer: function(e, t, a) {
        V = "useReducer", ee();
        var i = L.H;
        L.H = ku;
        try {
          return Qa(e, t, a);
        } finally {
          L.H = i;
        }
      },
      useRef: function() {
        return V = "useRef", ee(), ot().memoizedState;
      },
      useState: function() {
        V = "useState", ee();
        var e = L.H;
        L.H = ku;
        try {
          return Qa(dt);
        } finally {
          L.H = e;
        }
      },
      useDebugValue: function() {
        V = "useDebugValue", ee();
      },
      useDeferredValue: function(e, t) {
        return V = "useDeferredValue", ee(), kf(e, t);
      },
      useTransition: function() {
        return V = "useTransition", ee(), Gr();
      },
      useSyncExternalStore: function(e, t, a) {
        return V = "useSyncExternalStore", ee(), Vf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return V = "useId", ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return V = "useFormState", ee(), po(), xr(e);
      },
      useActionState: function(e) {
        return V = "useActionState", ee(), xr(e);
      },
      useOptimistic: function(e, t) {
        return V = "useOptimistic", ee(), _u(e, t);
      },
      useHostTransitionStatus: sa,
      useMemoCache: Pt,
      useCacheRefresh: function() {
        return V = "useCacheRefresh", ee(), ot().memoizedState;
      }
    }, T1 = {
      readContext: function(e) {
        return Ct(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return V = "useCallback", ee(), dc(e, t);
      },
      useContext: function(e) {
        return V = "useContext", ee(), Ct(e);
      },
      useEffect: function(e, t) {
        V = "useEffect", ee(), sl(2048, Bl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return V = "useImperativeHandle", ee(), Gn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return V = "useInsertionEffect", ee(), sl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return V = "useLayoutEffect", ee(), sl(4, la, e, t);
      },
      useMemo: function(e, t) {
        V = "useMemo", ee();
        var a = L.H;
        L.H = kv;
        try {
          return vi(e, t);
        } finally {
          L.H = a;
        }
      },
      useReducer: function(e, t, a) {
        V = "useReducer", ee();
        var i = L.H;
        L.H = kv;
        try {
          return rc(e, t, a);
        } finally {
          L.H = i;
        }
      },
      useRef: function() {
        return V = "useRef", ee(), ot().memoizedState;
      },
      useState: function() {
        V = "useState", ee();
        var e = L.H;
        L.H = kv;
        try {
          return rc(dt);
        } finally {
          L.H = e;
        }
      },
      useDebugValue: function() {
        V = "useDebugValue", ee();
      },
      useDeferredValue: function(e, t) {
        return V = "useDeferredValue", ee(), Br(e, t);
      },
      useTransition: function() {
        return V = "useTransition", ee(), Lr();
      },
      useSyncExternalStore: function(e, t, a) {
        return V = "useSyncExternalStore", ee(), Vf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return V = "useId", ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return V = "useFormState", ee(), po(), Ao(e);
      },
      useActionState: function(e) {
        return V = "useActionState", ee(), Ao(e);
      },
      useOptimistic: function(e, t) {
        return V = "useOptimistic", ee(), Cr(e, t);
      },
      useHostTransitionStatus: sa,
      useMemoCache: Pt,
      useCacheRefresh: function() {
        return V = "useCacheRefresh", ee(), ot().memoizedState;
      }
    }, Gi = {
      readContext: function(e) {
        return w(), Ct(e);
      },
      use: function(e) {
        return x(), jn(e);
      },
      useCallback: function(e, t) {
        return V = "useCallback", x(), $e(), Jf(e, t);
      },
      useContext: function(e) {
        return V = "useContext", x(), $e(), Ct(e);
      },
      useEffect: function(e, t) {
        return V = "useEffect", x(), $e(), Hr(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return V = "useImperativeHandle", x(), $e(), wr(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        V = "useInsertionEffect", x(), $e(), Ka(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return V = "useLayoutEffect", x(), $e(), Nr(e, t);
      },
      useMemo: function(e, t) {
        V = "useMemo", x(), $e();
        var a = L.H;
        L.H = Gi;
        try {
          return qr(e, t);
        } finally {
          L.H = a;
        }
      },
      useReducer: function(e, t, a) {
        V = "useReducer", x(), $e();
        var i = L.H;
        L.H = Gi;
        try {
          return rt(e, t, a);
        } finally {
          L.H = i;
        }
      },
      useRef: function(e) {
        return V = "useRef", x(), $e(), Kf(e);
      },
      useState: function(e) {
        V = "useState", x(), $e();
        var t = L.H;
        L.H = Gi;
        try {
          return Mu(e);
        } finally {
          L.H = t;
        }
      },
      useDebugValue: function() {
        V = "useDebugValue", x(), $e();
      },
      useDeferredValue: function(e, t) {
        return V = "useDeferredValue", x(), $e(), jr(e, t);
      },
      useTransition: function() {
        return V = "useTransition", x(), $e(), Ln();
      },
      useSyncExternalStore: function(e, t, a) {
        return V = "useSyncExternalStore", x(), $e(), zu(
          e,
          t,
          a
        );
      },
      useId: function() {
        return V = "useId", x(), $e(), Vn();
      },
      useFormState: function(e, t) {
        return V = "useFormState", x(), $e(), Eo(e, t);
      },
      useActionState: function(e, t) {
        return V = "useActionState", x(), $e(), Eo(e, t);
      },
      useOptimistic: function(e) {
        return V = "useOptimistic", x(), $e(), pn(e);
      },
      useMemoCache: function(e) {
        return x(), Pt(e);
      },
      useHostTransitionStatus: sa,
      useCacheRefresh: function() {
        return V = "useCacheRefresh", $e(), yc();
      }
    }, ku = {
      readContext: function(e) {
        return w(), Ct(e);
      },
      use: function(e) {
        return x(), jn(e);
      },
      useCallback: function(e, t) {
        return V = "useCallback", x(), ee(), dc(e, t);
      },
      useContext: function(e) {
        return V = "useContext", x(), ee(), Ct(e);
      },
      useEffect: function(e, t) {
        V = "useEffect", x(), ee(), sl(2048, Bl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return V = "useImperativeHandle", x(), ee(), Gn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return V = "useInsertionEffect", x(), ee(), sl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return V = "useLayoutEffect", x(), ee(), sl(4, la, e, t);
      },
      useMemo: function(e, t) {
        V = "useMemo", x(), ee();
        var a = L.H;
        L.H = ku;
        try {
          return vi(e, t);
        } finally {
          L.H = a;
        }
      },
      useReducer: function(e, t, a) {
        V = "useReducer", x(), ee();
        var i = L.H;
        L.H = ku;
        try {
          return Qa(e, t, a);
        } finally {
          L.H = i;
        }
      },
      useRef: function() {
        return V = "useRef", x(), ee(), ot().memoizedState;
      },
      useState: function() {
        V = "useState", x(), ee();
        var e = L.H;
        L.H = ku;
        try {
          return Qa(dt);
        } finally {
          L.H = e;
        }
      },
      useDebugValue: function() {
        V = "useDebugValue", x(), ee();
      },
      useDeferredValue: function(e, t) {
        return V = "useDeferredValue", x(), ee(), kf(e, t);
      },
      useTransition: function() {
        return V = "useTransition", x(), ee(), Gr();
      },
      useSyncExternalStore: function(e, t, a) {
        return V = "useSyncExternalStore", x(), ee(), Vf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return V = "useId", x(), ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return V = "useFormState", x(), ee(), xr(e);
      },
      useActionState: function(e) {
        return V = "useActionState", x(), ee(), xr(e);
      },
      useOptimistic: function(e, t) {
        return V = "useOptimistic", x(), ee(), _u(e, t);
      },
      useMemoCache: function(e) {
        return x(), Pt(e);
      },
      useHostTransitionStatus: sa,
      useCacheRefresh: function() {
        return V = "useCacheRefresh", ee(), ot().memoizedState;
      }
    }, kv = {
      readContext: function(e) {
        return w(), Ct(e);
      },
      use: function(e) {
        return x(), jn(e);
      },
      useCallback: function(e, t) {
        return V = "useCallback", x(), ee(), dc(e, t);
      },
      useContext: function(e) {
        return V = "useContext", x(), ee(), Ct(e);
      },
      useEffect: function(e, t) {
        V = "useEffect", x(), ee(), sl(2048, Bl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return V = "useImperativeHandle", x(), ee(), Gn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return V = "useInsertionEffect", x(), ee(), sl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return V = "useLayoutEffect", x(), ee(), sl(4, la, e, t);
      },
      useMemo: function(e, t) {
        V = "useMemo", x(), ee();
        var a = L.H;
        L.H = ku;
        try {
          return vi(e, t);
        } finally {
          L.H = a;
        }
      },
      useReducer: function(e, t, a) {
        V = "useReducer", x(), ee();
        var i = L.H;
        L.H = ku;
        try {
          return rc(e, t, a);
        } finally {
          L.H = i;
        }
      },
      useRef: function() {
        return V = "useRef", x(), ee(), ot().memoizedState;
      },
      useState: function() {
        V = "useState", x(), ee();
        var e = L.H;
        L.H = ku;
        try {
          return rc(dt);
        } finally {
          L.H = e;
        }
      },
      useDebugValue: function() {
        V = "useDebugValue", x(), ee();
      },
      useDeferredValue: function(e, t) {
        return V = "useDeferredValue", x(), ee(), Br(e, t);
      },
      useTransition: function() {
        return V = "useTransition", x(), ee(), Lr();
      },
      useSyncExternalStore: function(e, t, a) {
        return V = "useSyncExternalStore", x(), ee(), Vf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return V = "useId", x(), ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return V = "useFormState", x(), ee(), Ao(e);
      },
      useActionState: function(e) {
        return V = "useActionState", x(), ee(), Ao(e);
      },
      useOptimistic: function(e, t) {
        return V = "useOptimistic", x(), ee(), Cr(e, t);
      },
      useMemoCache: function(e) {
        return x(), Pt(e);
      },
      useHostTransitionStatus: sa,
      useCacheRefresh: function() {
        return V = "useCacheRefresh", ee(), ot().memoizedState;
      }
    };
    var E1 = {
      react_stack_bottom_frame: function(e, t, a) {
        var i = ba;
        ba = !0;
        try {
          return e(t, a);
        } finally {
          ba = i;
        }
      }
    }, i0 = E1.react_stack_bottom_frame.bind(E1), A1 = {
      react_stack_bottom_frame: function(e) {
        var t = ba;
        ba = !0;
        try {
          return e.render();
        } finally {
          ba = t;
        }
      }
    }, R1 = A1.react_stack_bottom_frame.bind(A1), O1 = {
      react_stack_bottom_frame: function(e, t) {
        try {
          t.componentDidMount();
        } catch (a) {
          Me(e, e.return, a);
        }
      }
    }, c0 = O1.react_stack_bottom_frame.bind(
      O1
    ), D1 = {
      react_stack_bottom_frame: function(e, t, a, i, o) {
        try {
          t.componentDidUpdate(a, i, o);
        } catch (f) {
          Me(e, e.return, f);
        }
      }
    }, z1 = D1.react_stack_bottom_frame.bind(
      D1
    ), M1 = {
      react_stack_bottom_frame: function(e, t) {
        var a = t.stack;
        e.componentDidCatch(t.value, {
          componentStack: a !== null ? a : ""
        });
      }
    }, LS = M1.react_stack_bottom_frame.bind(
      M1
    ), _1 = {
      react_stack_bottom_frame: function(e, t, a) {
        try {
          a.componentWillUnmount();
        } catch (i) {
          Me(e, t, i);
        }
      }
    }, U1 = _1.react_stack_bottom_frame.bind(
      _1
    ), C1 = {
      react_stack_bottom_frame: function(e) {
        e.resourceKind != null && console.error(
          "Expected only SimpleEffects when enableUseEffectCRUDOverload is disabled, got %s",
          e.resourceKind
        );
        var t = e.create;
        return e = e.inst, t = t(), e.destroy = t;
      }
    }, VS = C1.react_stack_bottom_frame.bind(C1), x1 = {
      react_stack_bottom_frame: function(e, t, a) {
        try {
          a();
        } catch (i) {
          Me(e, t, i);
        }
      }
    }, XS = x1.react_stack_bottom_frame.bind(x1), H1 = {
      react_stack_bottom_frame: function(e) {
        var t = e._init;
        return t(e._payload);
      }
    }, of = H1.react_stack_bottom_frame.bind(H1), mh = null, lp = 0, Fe = null, o0, N1 = o0 = !1, w1 = {}, q1 = {}, j1 = {};
    et = function(e, t, a) {
      if (a !== null && typeof a == "object" && a._store && (!a._store.validated && a.key == null || a._store.validated === 2)) {
        if (typeof a._store != "object")
          throw Error(
            "React Component in warnForMissingKey should have a _store. This error is likely caused by a bug in React. Please file an issue."
          );
        a._store.validated = 1;
        var i = re(e), o = i || "null";
        if (!w1[o]) {
          w1[o] = !0, a = a._owner, e = e._debugOwner;
          var f = "";
          e && typeof e.tag == "number" && (o = re(e)) && (f = `

Check the render method of \`` + o + "`."), f || i && (f = `

Check the top-level render call using <` + i + ">.");
          var d = "";
          a != null && e !== a && (i = null, typeof a.tag == "number" ? i = re(a) : typeof a.name == "string" && (i = a.name), i && (d = " It was passed a child from " + i + ".")), ye(t, function() {
            console.error(
              'Each child in a list should have a unique "key" prop.%s%s See https://react.dev/link/warning-keys for more information.',
              f,
              d
            );
          });
        }
      }
    };
    var ph = If(!0), B1 = If(!1), mu = Rt(null), Li = null, vh = 1, ap = 2, Yl = Rt(0), Y1 = {}, G1 = /* @__PURE__ */ new Set(), L1 = /* @__PURE__ */ new Set(), V1 = /* @__PURE__ */ new Set(), X1 = /* @__PURE__ */ new Set(), Q1 = /* @__PURE__ */ new Set(), Z1 = /* @__PURE__ */ new Set(), K1 = /* @__PURE__ */ new Set(), J1 = /* @__PURE__ */ new Set(), k1 = /* @__PURE__ */ new Set(), $1 = /* @__PURE__ */ new Set();
    Object.freeze(Y1);
    var f0 = {
      enqueueSetState: function(e, t, a) {
        e = e._reactInternals;
        var i = ha(e), o = wn(i);
        o.payload = t, a != null && (by(a), o.callback = a), t = hn(e, o, i), t !== null && (Kt(t, e, i), di(t, e, i)), zn(e, i);
      },
      enqueueReplaceState: function(e, t, a) {
        e = e._reactInternals;
        var i = ha(e), o = wn(i);
        o.tag = y1, o.payload = t, a != null && (by(a), o.callback = a), t = hn(e, o, i), t !== null && (Kt(t, e, i), di(t, e, i)), zn(e, i);
      },
      enqueueForceUpdate: function(e, t) {
        e = e._reactInternals;
        var a = ha(e), i = wn(a);
        i.tag = m1, t != null && (by(t), i.callback = t), t = hn(e, i, a), t !== null && (Kt(t, e, a), di(t, e, a)), fe !== null && typeof fe.markForceUpdateScheduled == "function" && fe.markForceUpdateScheduled(e, a);
      }
    }, s0 = typeof reportError == "function" ? reportError : function(e) {
      if (typeof window == "object" && typeof window.ErrorEvent == "function") {
        var t = new window.ErrorEvent("error", {
          bubbles: !0,
          cancelable: !0,
          message: typeof e == "object" && e !== null && typeof e.message == "string" ? String(e.message) : String(e),
          error: e
        });
        if (!window.dispatchEvent(t)) return;
      } else if (typeof It == "object" && typeof It.emit == "function") {
        It.emit("uncaughtException", e);
        return;
      }
      console.error(e);
    }, gh = null, r0 = null, W1 = Error(
      "This is not a real error. It's an implementation detail of React's selective hydration feature. If this leaks into userspace, it's a bug in React. Please file an issue."
    ), Kl = !1, F1 = {}, I1 = {}, P1 = {}, eb = {}, bh = !1, tb = {}, d0 = {}, h0 = {
      dehydrated: null,
      treeContext: null,
      retryLane: 0,
      hydrationErrors: null
    }, lb = !1, ab = null;
    ab = /* @__PURE__ */ new Set();
    var Zc = !1, hl = !1, y0 = !1, nb = typeof WeakSet == "function" ? WeakSet : Set, Jl = null, Sh = null, Th = null, Rl = null, ln = !1, $u = null, np = 8192, QS = {
      getCacheForType: function(e) {
        var t = Ct(jl), a = t.data.get(e);
        return a === void 0 && (a = e(), t.data.set(e, a)), a;
      },
      getOwner: function() {
        return xa;
      }
    };
    if (typeof Symbol == "function" && Symbol.for) {
      var up = Symbol.for;
      up("selector.component"), up("selector.has_pseudo_class"), up("selector.role"), up("selector.test_id"), up("selector.text");
    }
    var ZS = [], KS = typeof WeakMap == "function" ? WeakMap : Map, An = 0, qa = 2, Wu = 4, Kc = 0, ip = 1, Eh = 2, m0 = 3, ks = 4, $v = 6, ub = 5, Et = An, xt = null, at = null, ut = 0, an = 0, cp = 1, $s = 2, op = 3, ib = 4, p0 = 5, Ah = 6, fp = 7, v0 = 8, Ws = 9, Mt = an, Rn = null, ff = !1, Rh = !1, g0 = !1, Vi = 0, ul = Kc, sf = 0, rf = 0, b0 = 0, On = 0, Fs = 0, sp = null, ja = null, Wv = !1, S0 = 0, cb = 300, Fv = 1 / 0, ob = 500, rp = null, df = null, JS = 0, kS = 1, $S = 2, Is = 0, fb = 1, sb = 2, rb = 3, WS = 4, T0 = 5, aa = 0, hf = null, Oh = null, yf = 0, E0 = 0, A0 = null, db = null, FS = 50, dp = 0, R0 = null, O0 = !1, Iv = !1, IS = 50, Ps = 0, hp = null, Dh = !1, Pv = null, hb = !1, yb = /* @__PURE__ */ new Set(), PS = {}, eg = null, zh = null, D0 = !1, z0 = !1, tg = !1, M0 = !1, er = 0, _0 = {};
    (function() {
      for (var e = 0; e < Kg.length; e++) {
        var t = Kg[e], a = t.toLowerCase();
        t = t[0].toUpperCase() + t.slice(1), on(a, "on" + t);
      }
      on(I0, "onAnimationEnd"), on(P0, "onAnimationIteration"), on(e1, "onAnimationStart"), on("dblclick", "onDoubleClick"), on("focusin", "onFocus"), on("focusout", "onBlur"), on(US, "onTransitionRun"), on(CS, "onTransitionStart"), on(xS, "onTransitionCancel"), on(t1, "onTransitionEnd");
    })(), ue("onMouseEnter", ["mouseout", "mouseover"]), ue("onMouseLeave", ["mouseout", "mouseover"]), ue("onPointerEnter", ["pointerout", "pointerover"]), ue("onPointerLeave", ["pointerout", "pointerover"]), le(
      "onChange",
      "change click focusin focusout input keydown keyup selectionchange".split(
        " "
      )
    ), le(
      "onSelect",
      "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(
        " "
      )
    ), le("onBeforeInput", [
      "compositionend",
      "keypress",
      "textInput",
      "paste"
    ]), le(
      "onCompositionEnd",
      "compositionend focusout keydown keypress keyup mousedown".split(" ")
    ), le(
      "onCompositionStart",
      "compositionstart focusout keydown keypress keyup mousedown".split(" ")
    ), le(
      "onCompositionUpdate",
      "compositionupdate focusout keydown keypress keyup mousedown".split(" ")
    );
    var yp = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(
      " "
    ), U0 = new Set(
      "beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(yp)
    ), lg = "_reactListening" + Math.random().toString(36).slice(2), mb = !1, pb = !1, ag = !1, vb = !1, ng = !1, ug = !1, gb = !1, ig = {}, eT = /\r\n?/g, tT = /\u0000|\uFFFD/g, tr = "http://www.w3.org/1999/xlink", C0 = "http://www.w3.org/XML/1998/namespace", lT = "javascript:throw new Error('React form unexpectedly submitted.')", aT = "suppressHydrationWarning", cg = "$", og = "/$", Jc = "$?", mp = "$!", nT = 1, uT = 2, iT = 4, x0 = "F!", bb = "F", Sb = "complete", cT = "style", kc = 0, Mh = 1, fg = 2, H0 = null, N0 = null, Tb = { dialog: !0, webview: !0 }, w0 = null, Eb = typeof setTimeout == "function" ? setTimeout : void 0, oT = typeof clearTimeout == "function" ? clearTimeout : void 0, lr = -1, Ab = typeof Promise == "function" ? Promise : void 0, fT = typeof queueMicrotask == "function" ? queueMicrotask : typeof Ab < "u" ? function(e) {
      return Ab.resolve(null).then(e).catch(um);
    } : Eb, q0 = null, ar = 0, pp = 1, Rb = 2, Ob = 3, pu = 4, vu = /* @__PURE__ */ new Map(), Db = /* @__PURE__ */ new Set(), $c = Ce.d;
    Ce.d = {
      f: function() {
        var e = $c.f(), t = Rc();
        return e || t;
      },
      r: function(e) {
        var t = zl(e);
        t !== null && t.tag === 5 && t.type === "form" ? py(t) : $c.r(e);
      },
      D: function(e) {
        $c.D(e), dv("dns-prefetch", e, null);
      },
      C: function(e, t) {
        $c.C(e, t), dv("preconnect", e, t);
      },
      L: function(e, t, a) {
        $c.L(e, t, a);
        var i = _h;
        if (i && e && t) {
          var o = 'link[rel="preload"][as="' + Aa(t) + '"]';
          t === "image" && a && a.imageSrcSet ? (o += '[imagesrcset="' + Aa(
            a.imageSrcSet
          ) + '"]', typeof a.imageSizes == "string" && (o += '[imagesizes="' + Aa(
            a.imageSizes
          ) + '"]')) : o += '[href="' + Aa(e) + '"]';
          var f = o;
          switch (t) {
            case "style":
              f = zi(e);
              break;
            case "script":
              f = Cc(e);
          }
          vu.has(f) || (e = Je(
            {
              rel: "preload",
              href: t === "image" && a && a.imageSrcSet ? void 0 : e,
              as: t
            },
            a
          ), vu.set(f, e), i.querySelector(o) !== null || t === "style" && i.querySelector(
            lu(f)
          ) || t === "script" && i.querySelector(xc(f)) || (t = i.createElement("link"), kt(t, "link", e), D(t), i.head.appendChild(t)));
        }
      },
      m: function(e, t) {
        $c.m(e, t);
        var a = _h;
        if (a && e) {
          var i = t && typeof t.as == "string" ? t.as : "script", o = 'link[rel="modulepreload"][as="' + Aa(i) + '"][href="' + Aa(e) + '"]', f = o;
          switch (i) {
            case "audioworklet":
            case "paintworklet":
            case "serviceworker":
            case "sharedworker":
            case "worker":
            case "script":
              f = Cc(e);
          }
          if (!vu.has(f) && (e = Je({ rel: "modulepreload", href: e }, t), vu.set(f, e), a.querySelector(o) === null)) {
            switch (i) {
              case "audioworklet":
              case "paintworklet":
              case "serviceworker":
              case "sharedworker":
              case "worker":
              case "script":
                if (a.querySelector(xc(f)))
                  return;
            }
            i = a.createElement("link"), kt(i, "link", e), D(i), a.head.appendChild(i);
          }
        }
      },
      X: function(e, t) {
        $c.X(e, t);
        var a = _h;
        if (a && e) {
          var i = m(a).hoistableScripts, o = Cc(e), f = i.get(o);
          f || (f = a.querySelector(
            xc(o)
          ), f || (e = Je({ src: e, async: !0 }, t), (t = vu.get(o)) && hm(e, t), f = a.createElement("script"), D(f), kt(f, "link", e), a.head.appendChild(f)), f = {
            type: "script",
            instance: f,
            count: 1,
            state: null
          }, i.set(o, f));
        }
      },
      S: function(e, t, a) {
        $c.S(e, t, a);
        var i = _h;
        if (i && e) {
          var o = m(i).hoistableStyles, f = zi(e);
          t = t || "default";
          var d = o.get(f);
          if (!d) {
            var h = { loading: ar, preload: null };
            if (d = i.querySelector(
              lu(f)
            ))
              h.loading = pp | pu;
            else {
              e = Je(
                {
                  rel: "stylesheet",
                  href: e,
                  "data-precedence": t
                },
                a
              ), (a = vu.get(f)) && dm(e, a);
              var v = d = i.createElement("link");
              D(v), kt(v, "link", e), v._p = new Promise(function(b, B) {
                v.onload = b, v.onerror = B;
              }), v.addEventListener("load", function() {
                h.loading |= pp;
              }), v.addEventListener("error", function() {
                h.loading |= Rb;
              }), h.loading |= pu, xd(d, t, i);
            }
            d = {
              type: "stylesheet",
              instance: d,
              count: 1,
              state: h
            }, o.set(f, d);
          }
        }
      },
      M: function(e, t) {
        $c.M(e, t);
        var a = _h;
        if (a && e) {
          var i = m(a).hoistableScripts, o = Cc(e), f = i.get(o);
          f || (f = a.querySelector(
            xc(o)
          ), f || (e = Je({ src: e, async: !0, type: "module" }, t), (t = vu.get(o)) && hm(e, t), f = a.createElement("script"), D(f), kt(f, "link", e), a.head.appendChild(f)), f = {
            type: "script",
            instance: f,
            count: 1,
            state: null
          }, i.set(o, f));
        }
      }
    };
    var _h = typeof document > "u" ? null : document, sg = null, vp = null, j0 = null, rg = null, nr = wg, gp = {
      $$typeof: Ia,
      Provider: null,
      Consumer: null,
      _currentValue: nr,
      _currentValue2: nr,
      _threadCount: 0
    }, zb = "%c%s%c ", Mb = "background: #e6e6e6;background: light-dark(rgba(0,0,0,0.1), rgba(255,255,255,0.25));color: #000000;color: light-dark(#000000, #ffffff);border-radius: 2px", _b = "", dg = " ", sT = Function.prototype.bind, Ub = !1, Cb = null, xb = null, Hb = null, Nb = null, wb = null, qb = null, jb = null, Bb = null, Yb = null;
    Cb = function(e, t, a, i) {
      t = U(e, t), t !== null && (a = W(t.memoizedState, a, 0, i), t.memoizedState = a, t.baseState = a, e.memoizedProps = Je({}, e.memoizedProps), a = ia(e, 2), a !== null && Kt(a, e, 2));
    }, xb = function(e, t, a) {
      t = U(e, t), t !== null && (a = he(t.memoizedState, a, 0), t.memoizedState = a, t.baseState = a, e.memoizedProps = Je({}, e.memoizedProps), a = ia(e, 2), a !== null && Kt(a, e, 2));
    }, Hb = function(e, t, a, i) {
      t = U(e, t), t !== null && (a = ge(t.memoizedState, a, i), t.memoizedState = a, t.baseState = a, e.memoizedProps = Je({}, e.memoizedProps), a = ia(e, 2), a !== null && Kt(a, e, 2));
    }, Nb = function(e, t, a) {
      e.pendingProps = W(e.memoizedProps, t, 0, a), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ia(e, 2), t !== null && Kt(t, e, 2);
    }, wb = function(e, t) {
      e.pendingProps = he(e.memoizedProps, t, 0), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ia(e, 2), t !== null && Kt(t, e, 2);
    }, qb = function(e, t, a) {
      e.pendingProps = ge(
        e.memoizedProps,
        t,
        a
      ), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ia(e, 2), t !== null && Kt(t, e, 2);
    }, jb = function(e) {
      var t = ia(e, 2);
      t !== null && Kt(t, e, 2);
    }, Bb = function(e) {
      Qe = e;
    }, Yb = function(e) {
      Ee = e;
    };
    var hg = !0, yg = null, B0 = !1, mf = null, pf = null, vf = null, bp = /* @__PURE__ */ new Map(), Sp = /* @__PURE__ */ new Map(), gf = [], rT = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(
      " "
    ), mg = null;
    if (Os.prototype.render = Bd.prototype.render = function(e) {
      var t = this._internalRoot;
      if (t === null) throw Error("Cannot update an unmounted root.");
      var a = arguments;
      typeof a[1] == "function" ? console.error(
        "does not support the second callback argument. To execute a side effect after rendering, declare it in a component body with useEffect()."
      ) : ke(a[1]) ? console.error(
        "You passed a container to the second argument of root.render(...). You don't need to pass it again since you already passed it to create the root."
      ) : typeof a[1] < "u" && console.error(
        "You passed a second argument to root.render(...) but it only accepts one argument."
      ), a = e;
      var i = t.current, o = ha(i);
      Tt(i, o, a, t, null, null);
    }, Os.prototype.unmount = Bd.prototype.unmount = function() {
      var e = arguments;
      if (typeof e[0] == "function" && console.error(
        "does not support a callback argument. To execute a side effect after rendering, declare it in a component body with useEffect()."
      ), e = this._internalRoot, e !== null) {
        this._internalRoot = null;
        var t = e.containerInfo;
        (Et & (qa | Wu)) !== An && console.error(
          "Attempted to synchronously unmount a root while React was already rendering. React cannot finish unmounting the root until the current render has completed, which may lead to a race condition."
        ), Tt(e.current, 2, null, e, null, null), Rc(), t[wi] = null;
      }
    }, Os.prototype.unstable_scheduleHydration = function(e) {
      if (e) {
        var t = Tf();
        e = { blockedOn: null, target: e, priority: t };
        for (var a = 0; a < gf.length && t !== 0 && t < gf[a].priority; a++) ;
        gf.splice(a, 0, e), a === 0 && gv(e);
      }
    }, function() {
      var e = Ds.version;
      if (e !== "19.1.1")
        throw Error(
          `Incompatible React versions: The "react" and "react-dom" packages must have the exact same version. Instead got:
  - react:      ` + (e + `
  - react-dom:  19.1.1
Learn more: https://react.dev/warnings/version-mismatch`)
        );
    }(), typeof Map == "function" && Map.prototype != null && typeof Map.prototype.forEach == "function" && typeof Set == "function" && Set.prototype != null && typeof Set.prototype.clear == "function" && typeof Set.prototype.forEach == "function" || console.error(
      "React depends on Map and Set built-in types. Make sure that you load a polyfill in older browsers. https://react.dev/link/react-polyfills"
    ), Ce.findDOMNode = function(e) {
      var t = e._reactInternals;
      if (t === void 0)
        throw typeof e.render == "function" ? Error("Unable to find node on an unmounted component.") : (e = Object.keys(e).join(","), Error(
          "Argument appears to not be a ReactComponent. Keys: " + e
        ));
      return e = Nt(t), e = e !== null ? se(e) : null, e = e === null ? null : e.stateNode, e;
    }, !function() {
      var e = {
        bundleType: 1,
        version: "19.1.1",
        rendererPackageName: "react-dom",
        currentDispatcherRef: L,
        reconcilerVersion: "19.1.1"
      };
      return e.overrideHookState = Cb, e.overrideHookStateDeletePath = xb, e.overrideHookStateRenamePath = Hb, e.overrideProps = Nb, e.overridePropsDeletePath = wb, e.overridePropsRenamePath = qb, e.scheduleUpdate = jb, e.setErrorHandler = Bb, e.setSuspenseHandler = Yb, e.scheduleRefresh = je, e.scheduleRoot = ae, e.setRefreshHandler = At, e.getCurrentFiber = Cg, e.getLaneLabelMap = xg, e.injectProfilingHooks = il, Oe(e);
    }() && S && window.top === window.self && (-1 < navigator.userAgent.indexOf("Chrome") && navigator.userAgent.indexOf("Edge") === -1 || -1 < navigator.userAgent.indexOf("Firefox"))) {
      var Gb = window.location.protocol;
      /^(https?|file):$/.test(Gb) && console.info(
        "%cDownload the React DevTools for a better development experience: https://react.dev/link/react-devtools" + (Gb === "file:" ? `
You might need to use a local HTTP server (instead of file://): https://react.dev/link/react-devtools-faq` : ""),
        "font-weight:bold"
      );
    }
    Rp.createRoot = function(e, t) {
      if (!ke(e))
        throw Error("Target container is not a DOM element.");
      Tv(e);
      var a = !1, i = "", o = Sy, f = Wp, d = Zr, h = null;
      return t != null && (t.hydrate ? console.warn(
        "hydrate through createRoot is deprecated. Use ReactDOMClient.hydrateRoot(container, <App />) instead."
      ) : typeof t == "object" && t !== null && t.$$typeof === Ui && console.error(
        `You passed a JSX element to createRoot. You probably meant to call root.render instead. Example usage:

  let root = createRoot(domContainer);
  root.render(<App />);`
      ), t.unstable_strictMode === !0 && (a = !0), t.identifierPrefix !== void 0 && (i = t.identifierPrefix), t.onUncaughtError !== void 0 && (o = t.onUncaughtError), t.onCaughtError !== void 0 && (f = t.onCaughtError), t.onRecoverableError !== void 0 && (d = t.onRecoverableError), t.unstable_transitionCallbacks !== void 0 && (h = t.unstable_transitionCallbacks)), t = pm(
        e,
        1,
        !1,
        null,
        null,
        a,
        i,
        o,
        f,
        d,
        h,
        null
      ), e[wi] = t.current, Iy(e), new Bd(t);
    }, Rp.hydrateRoot = function(e, t, a) {
      if (!ke(e))
        throw Error("Target container is not a DOM element.");
      Tv(e), t === void 0 && console.error(
        "Must provide initial children as second argument to hydrateRoot. Example usage: hydrateRoot(domContainer, <App />)"
      );
      var i = !1, o = "", f = Sy, d = Wp, h = Zr, v = null, b = null;
      return a != null && (a.unstable_strictMode === !0 && (i = !0), a.identifierPrefix !== void 0 && (o = a.identifierPrefix), a.onUncaughtError !== void 0 && (f = a.onUncaughtError), a.onCaughtError !== void 0 && (d = a.onCaughtError), a.onRecoverableError !== void 0 && (h = a.onRecoverableError), a.unstable_transitionCallbacks !== void 0 && (v = a.unstable_transitionCallbacks), a.formState !== void 0 && (b = a.formState)), t = pm(
        e,
        1,
        !0,
        t,
        a ?? null,
        i,
        o,
        f,
        d,
        h,
        v,
        b
      ), t.context = vm(null), a = t.current, i = ha(a), i = Ol(i), o = wn(i), o.callback = null, hn(a, o, i), a = i, t.current.lanes = a, gu(t, a), $a(t), e[wi] = t.current, Iy(e), new Os(t);
    }, Rp.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  }()), Rp;
}
var tS;
function MT() {
  if (tS) return gg.exports;
  tS = 1;
  function U() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) {
      if (It.env.NODE_ENV !== "production")
        throw new Error("^_^");
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(U);
      } catch (W) {
        console.error(W);
      }
    }
  }
  return It.env.NODE_ENV === "production" ? (U(), gg.exports = DT()) : gg.exports = zT(), gg.exports;
}
var _T = MT(), Fu = Ch();
const UT = ({
  evalStructure: U,
  selectedEvals: W,
  onEvalToggle: ge,
  showPrerequisites: _
}) => {
  const [he, Ee] = Fu.useState(
    new Set(U.map((w) => w.name))
  ), Qe = (w) => {
    Ee((te) => {
      const G = new Set(te);
      return G.has(w) ? G.delete(w) : G.add(w), G;
    });
  }, et = (w) => !_ || w.length === 0 ? null : /* @__PURE__ */ ie.jsxs("div", { className: "prerequisites", children: [
    /* @__PURE__ */ ie.jsx("span", { className: "prereq-label", children: "Requires:" }),
    w.map((te, G) => /* @__PURE__ */ ie.jsx("span", { className: "prereq-item", children: te.split("/").pop() }, G))
  ] }), x = (w) => null;
  return /* @__PURE__ */ ie.jsx("div", { className: "eval-category-view", children: U.map((w) => /* @__PURE__ */ ie.jsxs("div", { className: "category-section", children: [
    /* @__PURE__ */ ie.jsxs(
      "div",
      {
        className: "category-header",
        onClick: () => Qe(w.name),
        children: [
          /* @__PURE__ */ ie.jsx("span", { className: "expand-icon", children: he.has(w.name) ? "▼" : "▶" }),
          /* @__PURE__ */ ie.jsx("h4", { children: w.name }),
          /* @__PURE__ */ ie.jsxs("span", { className: "eval-count", children: [
            "(",
            w.children.length,
            " evals)"
          ] })
        ]
      }
    ),
    he.has(w.name) && /* @__PURE__ */ ie.jsx("div", { className: "eval-list", children: w.children.map((te) => {
      if (!te.eval_metadata) return null;
      const G = te.eval_metadata, z = W.includes(G.name);
      return /* @__PURE__ */ ie.jsxs(
        "div",
        {
          className: `eval-item ${z ? "selected" : ""}`,
          onClick: () => ge(G.name),
          children: [
            /* @__PURE__ */ ie.jsxs("div", { className: "eval-header", children: [
              /* @__PURE__ */ ie.jsx(
                "input",
                {
                  type: "checkbox",
                  checked: z,
                  onChange: () => ge(G.name),
                  onClick: (ae) => ae.stopPropagation()
                }
              ),
              /* @__PURE__ */ ie.jsx("span", { className: "eval-name", children: te.name }),
              x()
            ] }),
            G.description && /* @__PURE__ */ ie.jsx("div", { className: "eval-description", children: G.description }),
            et(G.prerequisites)
          ]
        },
        G.name
      );
    }) })
  ] }, w.name)) });
}, CT = ({
  evaluations: U,
  selectedEvals: W,
  onEvalToggle: ge,
  viewMode: _,
  showPrerequisites: he
}) => {
  const Ee = (x) => {
    const w = {
      easy: "#22c55e",
      medium: "#f59e0b",
      hard: "#ef4444"
    };
    return /* @__PURE__ */ ie.jsx(
      "span",
      {
        className: "difficulty-badge",
        style: { backgroundColor: w[x] },
        children: x
      }
    );
  }, Qe = (x) => !he || x.length === 0 ? null : /* @__PURE__ */ ie.jsxs("div", { className: "prerequisites", children: [
    /* @__PURE__ */ ie.jsx("span", { className: "prereq-label", children: "Requires:" }),
    x.map((w, te) => /* @__PURE__ */ ie.jsx("span", { className: "prereq-item", children: w.split("/").pop() }, te))
  ] }), et = (x) => x.includes("any") ? null : /* @__PURE__ */ ie.jsx("div", { className: "agent-requirements", children: x.map((w, te) => /* @__PURE__ */ ie.jsx("span", { className: "agent-req-badge", children: w }, te)) });
  if (_ === "category") {
    const x = U.reduce((w, te) => (w[te.category] || (w[te.category] = []), w[te.category].push(te), w), {});
    return /* @__PURE__ */ ie.jsx("div", { className: "eval-list-view category-view", children: Object.entries(x).map(([w, te]) => /* @__PURE__ */ ie.jsxs("div", { className: "category-section", children: [
      /* @__PURE__ */ ie.jsx("h4", { className: "category-title", children: w }),
      /* @__PURE__ */ ie.jsx("div", { className: "eval-grid", children: te.map((G) => {
        const z = W.includes(G.name);
        return /* @__PURE__ */ ie.jsxs(
          "div",
          {
            className: `eval-card ${z ? "selected" : ""}`,
            onClick: () => ge(G.name),
            children: [
              /* @__PURE__ */ ie.jsxs("div", { className: "eval-header", children: [
                /* @__PURE__ */ ie.jsx(
                  "input",
                  {
                    type: "checkbox",
                    checked: z,
                    onChange: () => ge(G.name),
                    onClick: (ae) => ae.stopPropagation()
                  }
                ),
                /* @__PURE__ */ ie.jsx("span", { className: "eval-name", children: G.name.split("/").pop() }),
                Ee(G.difficulty)
              ] }),
              et(G.agent_requirements),
              G.description && /* @__PURE__ */ ie.jsx("div", { className: "eval-description", children: G.description }),
              Qe(G.prerequisites)
            ]
          },
          G.name
        );
      }) })
    ] }, w)) });
  }
  return /* @__PURE__ */ ie.jsx("div", { className: "eval-list-view", children: U.map((x) => {
    const w = W.includes(x.name);
    return /* @__PURE__ */ ie.jsxs(
      "div",
      {
        className: `eval-item ${w ? "selected" : ""}`,
        onClick: () => ge(x.name),
        children: [
          /* @__PURE__ */ ie.jsxs("div", { className: "eval-header", children: [
            /* @__PURE__ */ ie.jsx(
              "input",
              {
                type: "checkbox",
                checked: w,
                onChange: () => ge(x.name),
                onClick: (te) => te.stopPropagation()
              }
            ),
            /* @__PURE__ */ ie.jsx("span", { className: "eval-name", children: x.name }),
            Ee(x.difficulty),
            et(x.agent_requirements)
          ] }),
          x.description && /* @__PURE__ */ ie.jsx("div", { className: "eval-description", children: x.description }),
          Qe(x.prerequisites)
        ]
      },
      x.name
    );
  }) });
}, xT = ({
  categoryFilter: U,
  availableCategories: W,
  onFilterChange: ge
}) => {
  const _ = (he) => {
    const Ee = U.includes(he) ? U.filter((Qe) => Qe !== he) : [...U, he];
    ge({ category: Ee });
  };
  return /* @__PURE__ */ ie.jsx("div", { className: "filter-panel", children: /* @__PURE__ */ ie.jsxs("div", { className: "filter-section", children: [
    /* @__PURE__ */ ie.jsx("label", { className: "filter-label", children: "Categories:" }),
    /* @__PURE__ */ ie.jsx("div", { className: "checkbox-group", children: W.map((he) => /* @__PURE__ */ ie.jsxs("label", { className: "checkbox-item", children: [
      /* @__PURE__ */ ie.jsx(
        "input",
        {
          type: "checkbox",
          checked: U.length === 0 || U.includes(he),
          onChange: () => _(he)
        }
      ),
      /* @__PURE__ */ ie.jsx("span", { className: "category-label", children: he.replace("_", " ") })
    ] }, he)) })
  ] }) });
}, HT = ({
  searchTerm: U,
  onSearchChange: W
}) => /* @__PURE__ */ ie.jsx("div", { className: "search-bar", children: /* @__PURE__ */ ie.jsxs("div", { className: "search-input-container", children: [
  /* @__PURE__ */ ie.jsx("span", { className: "search-icon", children: "🔍" }),
  /* @__PURE__ */ ie.jsx(
    "input",
    {
      type: "text",
      placeholder: "Search evaluations by name, description, or tags...",
      value: U,
      onChange: (ge) => W(ge.target.value),
      className: "search-input"
    }
  ),
  U && /* @__PURE__ */ ie.jsx(
    "button",
    {
      className: "clear-search",
      onClick: () => W(""),
      children: "✕"
    }
  )
] }) }), NT = ({ model: U }) => {
  var it, Nt;
  const [W, ge] = Fu.useState(null), [_, he] = Fu.useState([]), [Ee, Qe] = Fu.useState([]), [et, x] = Fu.useState("tree"), [w, te] = Fu.useState(""), [G, z] = Fu.useState(!0);
  Fu.useEffect(() => {
    const se = () => {
      const Ue = U.get("eval_data");
      Ue && (console.log("🔍 EvalFinder: Received eval data:", Ue), ge(Ue));
      const De = U.get("selected_evals");
      De && JSON.stringify(De) !== JSON.stringify(_) && he(De), Qe(U.get("category_filter") || []), x(U.get("view_mode") || "tree"), te(U.get("search_term") || ""), z(U.get("show_prerequisites") !== !1);
    };
    return se(), U.on("change", se), () => U.off("change", se);
  }, [U]), Fu.useEffect(() => {
    const se = U.get("selected_evals") || [];
    JSON.stringify(_) !== JSON.stringify(se) && (U.set("selected_evals", _), U.save_changes(), U.set("selection_changed", {
      selected_evals: _,
      action: "updated",
      timestamp: Date.now()
    }), U.save_changes());
  }, [_, U]);
  const ae = Fu.useMemo(() => {
    if (!(W != null && W.evaluations))
      return console.log("🔍 EvalFinder: No eval data or evaluations found", W), [];
    console.log("🔍 EvalFinder: Filtering", W.evaluations.length, "evaluations");
    const se = W.evaluations.filter((Ue) => {
      var De, bt;
      if (w) {
        const re = w.toLowerCase(), Rt = Ue.name.toLowerCase().includes(re), Se = ((De = Ue.description) == null ? void 0 : De.toLowerCase().includes(re)) || !1, ze = ((bt = Ue.tags) == null ? void 0 : bt.some((Ot) => Ot.toLowerCase().includes(re))) || !1;
        if (!Rt && !Se && !ze)
          return !1;
      }
      return !(Ee.length > 0 && !Ee.includes(Ue.category));
    });
    return console.log("🔍 EvalFinder: Filtered to", se.length, "evaluations"), se;
  }, [W, w, Ee]), je = Fu.useMemo(() => {
    if (!(W != null && W.categories)) return [];
    const se = new Set(ae.map((Ue) => Ue.name));
    return W.categories.map((Ue) => ({
      ...Ue,
      children: Ue.children.filter(
        (De) => De.eval_metadata && se.has(De.eval_metadata.name)
      )
    })).filter((Ue) => Ue.children.length > 0);
  }, [W, ae]), At = (se) => {
    he(
      (Ue) => Ue.includes(se) ? Ue.filter((De) => De !== se) : [...Ue, se]
    );
  }, ke = () => {
    const se = ae.map((Ue) => Ue.name);
    he(se);
  }, nt = () => {
    he([]);
  }, el = (se) => {
    se.category !== void 0 && (Qe(se.category), U.set("category_filter", se.category)), U.save_changes();
  };
  return W ? /* @__PURE__ */ ie.jsxs("div", { className: "eval-finder-container", children: [
    /* @__PURE__ */ ie.jsxs("div", { className: "eval-finder-header", children: [
      /* @__PURE__ */ ie.jsx("h3", { children: "🎯 Evaluation Finder" }),
      /* @__PURE__ */ ie.jsxs("div", { className: "stats", children: [
        _.length,
        " of ",
        ae.length,
        " selected"
      ] })
    ] }),
    /* @__PURE__ */ ie.jsx(
      HT,
      {
        searchTerm: w,
        onSearchChange: te
      }
    ),
    /* @__PURE__ */ ie.jsx(
      xT,
      {
        categoryFilter: Ee,
        availableCategories: ((it = W.categories) == null ? void 0 : it.map((se) => se.name)) || [],
        onFilterChange: el
      }
    ),
    /* @__PURE__ */ ie.jsxs("div", { className: "view-controls", children: [
      /* @__PURE__ */ ie.jsxs("div", { className: "view-mode-selector", children: [
        /* @__PURE__ */ ie.jsx("label", { children: "View:" }),
        /* @__PURE__ */ ie.jsxs(
          "select",
          {
            value: et,
            onChange: (se) => x(se.target.value),
            children: [
              /* @__PURE__ */ ie.jsx("option", { value: "tree", children: "Tree" }),
              /* @__PURE__ */ ie.jsx("option", { value: "list", children: "List" }),
              /* @__PURE__ */ ie.jsx("option", { value: "category", children: "By Category" })
            ]
          }
        )
      ] }),
      /* @__PURE__ */ ie.jsxs("div", { className: "selection-controls", children: [
        /* @__PURE__ */ ie.jsxs(
          "button",
          {
            onClick: ke,
            className: "btn-select-all",
            disabled: _.length === ae.length,
            title: `Select all ${ae.length} evaluations`,
            children: [
              "📋 Select All (",
              ae.length,
              ")"
            ]
          }
        ),
        /* @__PURE__ */ ie.jsxs(
          "button",
          {
            onClick: nt,
            className: "btn-clear-all",
            disabled: _.length === 0,
            title: "Clear all selections",
            children: [
              "🗑️ Clear (",
              _.length,
              ")"
            ]
          }
        )
      ] })
    ] }),
    /* @__PURE__ */ ie.jsx("div", { className: "eval-content", children: ae.length === 0 ? /* @__PURE__ */ ie.jsxs("div", { className: "no-evals-message", children: [
      /* @__PURE__ */ ie.jsx("h4", { children: "No evaluations found" }),
      ((Nt = W == null ? void 0 : W.evaluations) == null ? void 0 : Nt.length) > 0 ? /* @__PURE__ */ ie.jsx("p", { children: "No evaluations match your current filters. Try adjusting your search or category filters." }) : /* @__PURE__ */ ie.jsxs("div", { children: [
        /* @__PURE__ */ ie.jsx("p", { children: "No evaluation data available." }),
        (W == null ? void 0 : W.has_policy_context) && /* @__PURE__ */ ie.jsxs("div", { className: "eval-counts", children: [
          /* @__PURE__ */ ie.jsxs("p", { children: [
            "📊 ",
            /* @__PURE__ */ ie.jsx("strong", { children: "Total evaluations found:" }),
            " ",
            (W == null ? void 0 : W.total_count) || 0
          ] }),
          /* @__PURE__ */ ie.jsx("p", { children: "These are evaluations that have been run on your selected policies." })
        ] })
      ] })
    ] }) : /* @__PURE__ */ ie.jsx(ie.Fragment, { children: et === "tree" ? /* @__PURE__ */ ie.jsx(
      UT,
      {
        evalStructure: je,
        selectedEvals: _,
        onEvalToggle: At,
        showPrerequisites: G
      }
    ) : /* @__PURE__ */ ie.jsx(
      CT,
      {
        evaluations: ae,
        selectedEvals: _,
        onEvalToggle: At,
        viewMode: et,
        showPrerequisites: G
      }
    ) }) })
  ] }) : /* @__PURE__ */ ie.jsx("div", { className: "eval-finder-container", children: /* @__PURE__ */ ie.jsx("div", { className: "loading-message", children: "🔍 Loading evaluations..." }) });
};
function wT({ model: U, el: W }) {
  const ge = _T.createRoot(W);
  return ge.render(/* @__PURE__ */ ie.jsx(NT, { model: U })), () => ge.unmount();
}
const qT = { render: wT };
export {
  qT as default
};
