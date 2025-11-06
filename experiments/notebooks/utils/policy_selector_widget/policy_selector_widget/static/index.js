function mT(H) {
  return H && H.__esModule && Object.prototype.hasOwnProperty.call(H, "default") ? H.default : H;
}
var aS = { exports: {} }, pl = aS.exports = {}, Qi, Zi;
function V0() {
  throw new Error("setTimeout has not been defined");
}
function X0() {
  throw new Error("clearTimeout has not been defined");
}
(function() {
  try {
    typeof setTimeout == "function" ? Qi = setTimeout : Qi = V0;
  } catch {
    Qi = V0;
  }
  try {
    typeof clearTimeout == "function" ? Zi = clearTimeout : Zi = X0;
  } catch {
    Zi = X0;
  }
})();
function nS(H) {
  if (Qi === setTimeout)
    return setTimeout(H, 0);
  if ((Qi === V0 || !Qi) && setTimeout)
    return Qi = setTimeout, setTimeout(H, 0);
  try {
    return Qi(H, 0);
  } catch {
    try {
      return Qi.call(null, H, 0);
    } catch {
      return Qi.call(this, H, 0);
    }
  }
}
function pT(H) {
  if (Zi === clearTimeout)
    return clearTimeout(H);
  if ((Zi === X0 || !Zi) && clearTimeout)
    return Zi = clearTimeout, clearTimeout(H);
  try {
    return Zi(H);
  } catch {
    try {
      return Zi.call(null, H);
    } catch {
      return Zi.call(this, H);
    }
  }
}
var Ic = [], Ch = !1, is, Eg = -1;
function vT() {
  !Ch || !is || (Ch = !1, is.length ? Ic = is.concat(Ic) : Eg = -1, Ic.length && uS());
}
function uS() {
  if (!Ch) {
    var H = nS(vT);
    Ch = !0;
    for (var F = Ic.length; F; ) {
      for (is = Ic, Ic = []; ++Eg < F; )
        is && is[Eg].run();
      Eg = -1, F = Ic.length;
    }
    is = null, Ch = !1, pT(H);
  }
}
pl.nextTick = function(H) {
  var F = new Array(arguments.length - 1);
  if (arguments.length > 1)
    for (var Re = 1; Re < arguments.length; Re++)
      F[Re - 1] = arguments[Re];
  Ic.push(new iS(H, F)), Ic.length === 1 && !Ch && nS(uS);
};
function iS(H, F) {
  this.fun = H, this.array = F;
}
iS.prototype.run = function() {
  this.fun.apply(null, this.array);
};
pl.title = "browser";
pl.browser = !0;
pl.env = {};
pl.argv = [];
pl.version = "";
pl.versions = {};
function Pc() {
}
pl.on = Pc;
pl.addListener = Pc;
pl.once = Pc;
pl.off = Pc;
pl.removeListener = Pc;
pl.removeAllListeners = Pc;
pl.emit = Pc;
pl.prependListener = Pc;
pl.prependOnceListener = Pc;
pl.listeners = function(H) {
  return [];
};
pl.binding = function(H) {
  throw new Error("process.binding is not supported");
};
pl.cwd = function() {
  return "/";
};
pl.chdir = function(H) {
  throw new Error("process.chdir is not supported");
};
pl.umask = function() {
  return 0;
};
var gT = aS.exports;
const Pt = /* @__PURE__ */ mT(gT);
var vg = { exports: {} }, Ep = {};
/**
 * @license React
 * react-jsx-runtime.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var Vb;
function bT() {
  if (Vb) return Ep;
  Vb = 1;
  var H = Symbol.for("react.transitional.element"), F = Symbol.for("react.fragment");
  function Re(_, re, Ae) {
    var Ne = null;
    if (Ae !== void 0 && (Ne = "" + Ae), re.key !== void 0 && (Ne = "" + re.key), "key" in re) {
      Ae = {};
      for (var st in re)
        st !== "key" && (Ae[st] = re[st]);
    } else Ae = re;
    return re = Ae.ref, {
      $$typeof: H,
      type: _,
      key: Ne,
      ref: re !== void 0 ? re : null,
      props: Ae
    };
  }
  return Ep.Fragment = F, Ep.jsx = Re, Ep.jsxs = Re, Ep;
}
var Rp = {}, gg = { exports: {} }, Ie = {}, Xb;
function ST() {
  if (Xb) return Ie;
  Xb = 1;
  var H = Symbol.for("react.transitional.element"), F = Symbol.for("react.portal"), Re = Symbol.for("react.fragment"), _ = Symbol.for("react.strict_mode"), re = Symbol.for("react.profiler"), Ae = Symbol.for("react.consumer"), Ne = Symbol.for("react.context"), st = Symbol.for("react.forward_ref"), j = Symbol.for("react.suspense"), k = Symbol.for("react.memo"), ie = Symbol.for("react.lazy"), K = Symbol.iterator;
  function D(g) {
    return g === null || typeof g != "object" ? null : (g = K && g[K] || g["@@iterator"], typeof g == "function" ? g : null);
  }
  var ue = {
    isMounted: function() {
      return !1;
    },
    enqueueForceUpdate: function() {
    },
    enqueueReplaceState: function() {
    },
    enqueueSetState: function() {
    }
  }, Oe = Object.assign, ot = {};
  function He(g, w, J) {
    this.props = g, this.context = w, this.refs = ot, this.updater = J || ue;
  }
  He.prototype.isReactComponent = {}, He.prototype.setState = function(g, w) {
    if (typeof g != "object" && typeof g != "function" && g != null)
      throw Error(
        "takes an object of state variables to update or a function which returns an object of state variables."
      );
    this.updater.enqueueSetState(this, g, w, "setState");
  }, He.prototype.forceUpdate = function(g) {
    this.updater.enqueueForceUpdate(this, g, "forceUpdate");
  };
  function Pe() {
  }
  Pe.prototype = He.prototype;
  function Ct(g, w, J) {
    this.props = g, this.context = w, this.refs = ot, this.updater = J || ue;
  }
  var Ke = Ct.prototype = new Pe();
  Ke.constructor = Ct, Oe(Ke, He.prototype), Ke.isPureReactComponent = !0;
  var At = Array.isArray, be = { H: null, A: null, T: null, S: null, V: null }, pt = Object.prototype.hasOwnProperty;
  function je(g, w, J, P, ce, De) {
    return J = De.ref, {
      $$typeof: H,
      type: g,
      key: w,
      ref: J !== void 0 ? J : null,
      props: De
    };
  }
  function St(g, w) {
    return je(
      g.type,
      w,
      void 0,
      void 0,
      void 0,
      g.props
    );
  }
  function de(g) {
    return typeof g == "object" && g !== null && g.$$typeof === H;
  }
  function Ot(g) {
    var w = { "=": "=0", ":": "=2" };
    return "$" + g.replace(/[=:]/g, function(J) {
      return w[J];
    });
  }
  var ve = /\/+/g;
  function ze(g, w) {
    return typeof g == "object" && g !== null && g.key != null ? Ot("" + g.key) : w.toString(36);
  }
  function Dt() {
  }
  function Ht(g) {
    switch (g.status) {
      case "fulfilled":
        return g.value;
      case "rejected":
        throw g.reason;
      default:
        switch (typeof g.status == "string" ? g.then(Dt, Dt) : (g.status = "pending", g.then(
          function(w) {
            g.status === "pending" && (g.status = "fulfilled", g.value = w);
          },
          function(w) {
            g.status === "pending" && (g.status = "rejected", g.reason = w);
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
  function le(g, w, J, P, ce) {
    var De = typeof g;
    (De === "undefined" || De === "boolean") && (g = null);
    var oe = !1;
    if (g === null) oe = !0;
    else
      switch (De) {
        case "bigint":
        case "string":
        case "number":
          oe = !0;
          break;
        case "object":
          switch (g.$$typeof) {
            case H:
            case F:
              oe = !0;
              break;
            case ie:
              return oe = g._init, le(
                oe(g._payload),
                w,
                J,
                P,
                ce
              );
          }
      }
    if (oe)
      return ce = ce(g), oe = P === "" ? "." + ze(g, 0) : P, At(ce) ? (J = "", oe != null && (J = oe.replace(ve, "$&/") + "/"), le(ce, w, J, "", function(Bt) {
        return Bt;
      })) : ce != null && (de(ce) && (ce = St(
        ce,
        J + (ce.key == null || g && g.key === ce.key ? "" : ("" + ce.key).replace(
          ve,
          "$&/"
        ) + "/") + oe
      )), w.push(ce)), 1;
    oe = 0;
    var cl = P === "" ? "." : P + ":";
    if (At(g))
      for (var xe = 0; xe < g.length; xe++)
        P = g[xe], De = cl + ze(P, xe), oe += le(
          P,
          w,
          J,
          De,
          ce
        );
    else if (xe = D(g), typeof xe == "function")
      for (g = xe.call(g), xe = 0; !(P = g.next()).done; )
        P = P.value, De = cl + ze(P, xe++), oe += le(
          P,
          w,
          J,
          De,
          ce
        );
    else if (De === "object") {
      if (typeof g.then == "function")
        return le(
          Ht(g),
          w,
          J,
          P,
          ce
        );
      throw w = String(g), Error(
        "Objects are not valid as a React child (found: " + (w === "[object Object]" ? "object with keys {" + Object.keys(g).join(", ") + "}" : w) + "). If you meant to render a collection of children, use an array instead."
      );
    }
    return oe;
  }
  function R(g, w, J) {
    if (g == null) return g;
    var P = [], ce = 0;
    return le(g, P, "", "", function(De) {
      return w.call(J, De, ce++);
    }), P;
  }
  function X(g) {
    if (g._status === -1) {
      var w = g._result;
      w = w(), w.then(
        function(J) {
          (g._status === 0 || g._status === -1) && (g._status = 1, g._result = J);
        },
        function(J) {
          (g._status === 0 || g._status === -1) && (g._status = 2, g._result = J);
        }
      ), g._status === -1 && (g._status = 0, g._result = w);
    }
    if (g._status === 1) return g._result.default;
    throw g._result;
  }
  var I = typeof reportError == "function" ? reportError : function(g) {
    if (typeof window == "object" && typeof window.ErrorEvent == "function") {
      var w = new window.ErrorEvent("error", {
        bubbles: !0,
        cancelable: !0,
        message: typeof g == "object" && g !== null && typeof g.message == "string" ? String(g.message) : String(g),
        error: g
      });
      if (!window.dispatchEvent(w)) return;
    } else if (typeof Pt == "object" && typeof Pt.emit == "function") {
      Pt.emit("uncaughtException", g);
      return;
    }
    console.error(g);
  };
  function ge() {
  }
  return Ie.Children = {
    map: R,
    forEach: function(g, w, J) {
      R(
        g,
        function() {
          w.apply(this, arguments);
        },
        J
      );
    },
    count: function(g) {
      var w = 0;
      return R(g, function() {
        w++;
      }), w;
    },
    toArray: function(g) {
      return R(g, function(w) {
        return w;
      }) || [];
    },
    only: function(g) {
      if (!de(g))
        throw Error(
          "React.Children.only expected to receive a single React element child."
        );
      return g;
    }
  }, Ie.Component = He, Ie.Fragment = Re, Ie.Profiler = re, Ie.PureComponent = Ct, Ie.StrictMode = _, Ie.Suspense = j, Ie.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = be, Ie.__COMPILER_RUNTIME = {
    __proto__: null,
    c: function(g) {
      return be.H.useMemoCache(g);
    }
  }, Ie.cache = function(g) {
    return function() {
      return g.apply(null, arguments);
    };
  }, Ie.cloneElement = function(g, w, J) {
    if (g == null)
      throw Error(
        "The argument must be a React element, but you passed " + g + "."
      );
    var P = Oe({}, g.props), ce = g.key, De = void 0;
    if (w != null)
      for (oe in w.ref !== void 0 && (De = void 0), w.key !== void 0 && (ce = "" + w.key), w)
        !pt.call(w, oe) || oe === "key" || oe === "__self" || oe === "__source" || oe === "ref" && w.ref === void 0 || (P[oe] = w[oe]);
    var oe = arguments.length - 2;
    if (oe === 1) P.children = J;
    else if (1 < oe) {
      for (var cl = Array(oe), xe = 0; xe < oe; xe++)
        cl[xe] = arguments[xe + 2];
      P.children = cl;
    }
    return je(g.type, ce, void 0, void 0, De, P);
  }, Ie.createContext = function(g) {
    return g = {
      $$typeof: Ne,
      _currentValue: g,
      _currentValue2: g,
      _threadCount: 0,
      Provider: null,
      Consumer: null
    }, g.Provider = g, g.Consumer = {
      $$typeof: Ae,
      _context: g
    }, g;
  }, Ie.createElement = function(g, w, J) {
    var P, ce = {}, De = null;
    if (w != null)
      for (P in w.key !== void 0 && (De = "" + w.key), w)
        pt.call(w, P) && P !== "key" && P !== "__self" && P !== "__source" && (ce[P] = w[P]);
    var oe = arguments.length - 2;
    if (oe === 1) ce.children = J;
    else if (1 < oe) {
      for (var cl = Array(oe), xe = 0; xe < oe; xe++)
        cl[xe] = arguments[xe + 2];
      ce.children = cl;
    }
    if (g && g.defaultProps)
      for (P in oe = g.defaultProps, oe)
        ce[P] === void 0 && (ce[P] = oe[P]);
    return je(g, De, void 0, void 0, null, ce);
  }, Ie.createRef = function() {
    return { current: null };
  }, Ie.forwardRef = function(g) {
    return { $$typeof: st, render: g };
  }, Ie.isValidElement = de, Ie.lazy = function(g) {
    return {
      $$typeof: ie,
      _payload: { _status: -1, _result: g },
      _init: X
    };
  }, Ie.memo = function(g, w) {
    return {
      $$typeof: k,
      type: g,
      compare: w === void 0 ? null : w
    };
  }, Ie.startTransition = function(g) {
    var w = be.T, J = {};
    be.T = J;
    try {
      var P = g(), ce = be.S;
      ce !== null && ce(J, P), typeof P == "object" && P !== null && typeof P.then == "function" && P.then(ge, I);
    } catch (De) {
      I(De);
    } finally {
      be.T = w;
    }
  }, Ie.unstable_useCacheRefresh = function() {
    return be.H.useCacheRefresh();
  }, Ie.use = function(g) {
    return be.H.use(g);
  }, Ie.useActionState = function(g, w, J) {
    return be.H.useActionState(g, w, J);
  }, Ie.useCallback = function(g, w) {
    return be.H.useCallback(g, w);
  }, Ie.useContext = function(g) {
    return be.H.useContext(g);
  }, Ie.useDebugValue = function() {
  }, Ie.useDeferredValue = function(g, w) {
    return be.H.useDeferredValue(g, w);
  }, Ie.useEffect = function(g, w, J) {
    var P = be.H;
    if (typeof J == "function")
      throw Error(
        "useEffect CRUD overload is not enabled in this build of React."
      );
    return P.useEffect(g, w);
  }, Ie.useId = function() {
    return be.H.useId();
  }, Ie.useImperativeHandle = function(g, w, J) {
    return be.H.useImperativeHandle(g, w, J);
  }, Ie.useInsertionEffect = function(g, w) {
    return be.H.useInsertionEffect(g, w);
  }, Ie.useLayoutEffect = function(g, w) {
    return be.H.useLayoutEffect(g, w);
  }, Ie.useMemo = function(g, w) {
    return be.H.useMemo(g, w);
  }, Ie.useOptimistic = function(g, w) {
    return be.H.useOptimistic(g, w);
  }, Ie.useReducer = function(g, w, J) {
    return be.H.useReducer(g, w, J);
  }, Ie.useRef = function(g) {
    return be.H.useRef(g);
  }, Ie.useState = function(g) {
    return be.H.useState(g);
  }, Ie.useSyncExternalStore = function(g, w, J) {
    return be.H.useSyncExternalStore(
      g,
      w,
      J
    );
  }, Ie.useTransition = function() {
    return be.H.useTransition();
  }, Ie.version = "19.1.1", Ie;
}
var Dp = { exports: {} };
Dp.exports;
var Qb;
function TT() {
  return Qb || (Qb = 1, function(H, F) {
    Pt.env.NODE_ENV !== "production" && function() {
      function Re(m, z) {
        Object.defineProperty(Ae.prototype, m, {
          get: function() {
            console.warn(
              "%s(...) is deprecated in plain JavaScript React classes. %s",
              z[0],
              z[1]
            );
          }
        });
      }
      function _(m) {
        return m === null || typeof m != "object" ? null : (m = Mn && m[Mn] || m["@@iterator"], typeof m == "function" ? m : null);
      }
      function re(m, z) {
        m = (m = m.constructor) && (m.displayName || m.name) || "ReactClass";
        var te = m + "." + z;
        Ki[te] || (console.error(
          "Can't call %s on a component that is not yet mounted. This is a no-op, but it might indicate a bug in your application. Instead, assign to `this.state` directly or define a `state = {};` class property with the desired state in the %s component.",
          z,
          m
        ), Ki[te] = !0);
      }
      function Ae(m, z, te) {
        this.props = m, this.context = z, this.refs = Sf, this.updater = te || _n;
      }
      function Ne() {
      }
      function st(m, z, te) {
        this.props = m, this.context = z, this.refs = Sf, this.updater = te || _n;
      }
      function j(m) {
        return "" + m;
      }
      function k(m) {
        try {
          j(m);
          var z = !1;
        } catch {
          z = !0;
        }
        if (z) {
          z = console;
          var te = z.error, ne = typeof Symbol == "function" && Symbol.toStringTag && m[Symbol.toStringTag] || m.constructor.name || "Object";
          return te.call(
            z,
            "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
            ne
          ), j(m);
        }
      }
      function ie(m) {
        if (m == null) return null;
        if (typeof m == "function")
          return m.$$typeof === cs ? null : m.displayName || m.name || null;
        if (typeof m == "string") return m;
        switch (m) {
          case g:
            return "Fragment";
          case J:
            return "Profiler";
          case w:
            return "StrictMode";
          case oe:
            return "Suspense";
          case cl:
            return "SuspenseList";
          case ua:
            return "Activity";
        }
        if (typeof m == "object")
          switch (typeof m.tag == "number" && console.error(
            "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
          ), m.$$typeof) {
            case ge:
              return "Portal";
            case ce:
              return (m.displayName || "Context") + ".Provider";
            case P:
              return (m._context.displayName || "Context") + ".Consumer";
            case De:
              var z = m.render;
              return m = m.displayName, m || (m = z.displayName || z.name || "", m = m !== "" ? "ForwardRef(" + m + ")" : "ForwardRef"), m;
            case xe:
              return z = m.displayName || null, z !== null ? z : ie(m.type) || "Memo";
            case Bt:
              z = m._payload, m = m._init;
              try {
                return ie(m(z));
              } catch {
              }
          }
        return null;
      }
      function K(m) {
        if (m === g) return "<>";
        if (typeof m == "object" && m !== null && m.$$typeof === Bt)
          return "<...>";
        try {
          var z = ie(m);
          return z ? "<" + z + ">" : "<...>";
        } catch {
          return "<...>";
        }
      }
      function D() {
        var m = Je.A;
        return m === null ? null : m.getOwner();
      }
      function ue() {
        return Error("react-stack-top-frame");
      }
      function Oe(m) {
        if (Un.call(m, "key")) {
          var z = Object.getOwnPropertyDescriptor(m, "key").get;
          if (z && z.isReactWarning) return !1;
        }
        return m.key !== void 0;
      }
      function ot(m, z) {
        function te() {
          Su || (Su = !0, console.error(
            "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
            z
          ));
        }
        te.isReactWarning = !0, Object.defineProperty(m, "key", {
          get: te,
          configurable: !0
        });
      }
      function He() {
        var m = ie(this.type);
        return Tf[m] || (Tf[m] = !0, console.error(
          "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
        )), m = this.props.ref, m !== void 0 ? m : null;
      }
      function Pe(m, z, te, ne, pe, we, Ge, ut) {
        return te = we.ref, m = {
          $$typeof: I,
          type: m,
          key: z,
          props: we,
          _owner: pe
        }, (te !== void 0 ? te : null) !== null ? Object.defineProperty(m, "ref", {
          enumerable: !1,
          get: He
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
          value: Ge
        }), Object.defineProperty(m, "_debugTask", {
          configurable: !1,
          enumerable: !1,
          writable: !0,
          value: ut
        }), Object.freeze && (Object.freeze(m.props), Object.freeze(m)), m;
      }
      function Ct(m, z) {
        return z = Pe(
          m.type,
          z,
          void 0,
          void 0,
          m._owner,
          m.props,
          m._debugStack,
          m._debugTask
        ), m._store && (z._store.validated = m._store.validated), z;
      }
      function Ke(m) {
        return typeof m == "object" && m !== null && m.$$typeof === I;
      }
      function At(m) {
        var z = { "=": "=0", ":": "=2" };
        return "$" + m.replace(/[=:]/g, function(te) {
          return z[te];
        });
      }
      function be(m, z) {
        return typeof m == "object" && m !== null && m.key != null ? (k(m.key), At("" + m.key)) : z.toString(36);
      }
      function pt() {
      }
      function je(m) {
        switch (m.status) {
          case "fulfilled":
            return m.value;
          case "rejected":
            throw m.reason;
          default:
            switch (typeof m.status == "string" ? m.then(pt, pt) : (m.status = "pending", m.then(
              function(z) {
                m.status === "pending" && (m.status = "fulfilled", m.value = z);
              },
              function(z) {
                m.status === "pending" && (m.status = "rejected", m.reason = z);
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
      function St(m, z, te, ne, pe) {
        var we = typeof m;
        (we === "undefined" || we === "boolean") && (m = null);
        var Ge = !1;
        if (m === null) Ge = !0;
        else
          switch (we) {
            case "bigint":
            case "string":
            case "number":
              Ge = !0;
              break;
            case "object":
              switch (m.$$typeof) {
                case I:
                case ge:
                  Ge = !0;
                  break;
                case Bt:
                  return Ge = m._init, St(
                    Ge(m._payload),
                    z,
                    te,
                    ne,
                    pe
                  );
              }
          }
        if (Ge) {
          Ge = m, pe = pe(Ge);
          var ut = ne === "" ? "." + be(Ge, 0) : ne;
          return Pu(pe) ? (te = "", ut != null && (te = ut.replace(zl, "$&/") + "/"), St(pe, z, te, "", function(al) {
            return al;
          })) : pe != null && (Ke(pe) && (pe.key != null && (Ge && Ge.key === pe.key || k(pe.key)), te = Ct(
            pe,
            te + (pe.key == null || Ge && Ge.key === pe.key ? "" : ("" + pe.key).replace(
              zl,
              "$&/"
            ) + "/") + ut
          ), ne !== "" && Ge != null && Ke(Ge) && Ge.key == null && Ge._store && !Ge._store.validated && (te._store.validated = 2), pe = te), z.push(pe)), 1;
        }
        if (Ge = 0, ut = ne === "" ? "." : ne + ":", Pu(m))
          for (var Ye = 0; Ye < m.length; Ye++)
            ne = m[Ye], we = ut + be(ne, Ye), Ge += St(
              ne,
              z,
              te,
              we,
              pe
            );
        else if (Ye = _(m), typeof Ye == "function")
          for (Ye === m.entries && (ja || console.warn(
            "Using Maps as children is not supported. Use an array of keyed ReactElements instead."
          ), ja = !0), m = Ye.call(m), Ye = 0; !(ne = m.next()).done; )
            ne = ne.value, we = ut + be(ne, Ye++), Ge += St(
              ne,
              z,
              te,
              we,
              pe
            );
        else if (we === "object") {
          if (typeof m.then == "function")
            return St(
              je(m),
              z,
              te,
              ne,
              pe
            );
          throw z = String(m), Error(
            "Objects are not valid as a React child (found: " + (z === "[object Object]" ? "object with keys {" + Object.keys(m).join(", ") + "}" : z) + "). If you meant to render a collection of children, use an array instead."
          );
        }
        return Ge;
      }
      function de(m, z, te) {
        if (m == null) return m;
        var ne = [], pe = 0;
        return St(m, ne, "", "", function(we) {
          return z.call(te, we, pe++);
        }), ne;
      }
      function Ot(m) {
        if (m._status === -1) {
          var z = m._result;
          z = z(), z.then(
            function(te) {
              (m._status === 0 || m._status === -1) && (m._status = 1, m._result = te);
            },
            function(te) {
              (m._status === 0 || m._status === -1) && (m._status = 2, m._result = te);
            }
          ), m._status === -1 && (m._status = 0, m._result = z);
        }
        if (m._status === 1)
          return z = m._result, z === void 0 && console.error(
            `lazy: Expected the result of a dynamic import() call. Instead received: %s

Your code should look like: 
  const MyComponent = lazy(() => import('./MyComponent'))

Did you accidentally put curly braces around the import?`,
            z
          ), "default" in z || console.error(
            `lazy: Expected the result of a dynamic import() call. Instead received: %s

Your code should look like: 
  const MyComponent = lazy(() => import('./MyComponent'))`,
            z
          ), z.default;
        throw m._result;
      }
      function ve() {
        var m = Je.H;
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
      function Dt(m) {
        if (ao === null)
          try {
            var z = ("require" + Math.random()).slice(0, 7);
            ao = (H && H[z]).call(
              H,
              "timers"
            ).setImmediate;
          } catch {
            ao = function(ne) {
              Ef === !1 && (Ef = !0, typeof MessageChannel > "u" && console.error(
                "This browser does not have a MessageChannel implementation, so enqueuing tasks via await act(async () => ...) will fail. Please file an issue at https://github.com/facebook/react/issues if you encounter this warning."
              ));
              var pe = new MessageChannel();
              pe.port1.onmessage = ne, pe.port2.postMessage(void 0);
            };
          }
        return ao(m);
      }
      function Ht(m) {
        return 1 < m.length && typeof AggregateError == "function" ? new AggregateError(m) : m[0];
      }
      function le(m, z) {
        z !== un - 1 && console.error(
          "You seem to have overlapping act() calls, this is not supported. Be sure to await previous act() calls before making a new one. "
        ), un = z;
      }
      function R(m, z, te) {
        var ne = Je.actQueue;
        if (ne !== null)
          if (ne.length !== 0)
            try {
              X(ne), Dt(function() {
                return R(m, z, te);
              });
              return;
            } catch (pe) {
              Je.thrownErrors.push(pe);
            }
          else Je.actQueue = null;
        0 < Je.thrownErrors.length ? (ne = Ht(Je.thrownErrors), Je.thrownErrors.length = 0, te(ne)) : z(m);
      }
      function X(m) {
        if (!Ml) {
          Ml = !0;
          var z = 0;
          try {
            for (; z < m.length; z++) {
              var te = m[z];
              do {
                Je.didUsePromise = !1;
                var ne = te(!1);
                if (ne !== null) {
                  if (Je.didUsePromise) {
                    m[z] = te, m.splice(0, z);
                    return;
                  }
                  te = ne;
                } else break;
              } while (!0);
            }
            m.length = 0;
          } catch (pe) {
            m.splice(0, z + 1), Je.thrownErrors.push(pe);
          } finally {
            Ml = !1;
          }
        }
      }
      typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
      var I = Symbol.for("react.transitional.element"), ge = Symbol.for("react.portal"), g = Symbol.for("react.fragment"), w = Symbol.for("react.strict_mode"), J = Symbol.for("react.profiler"), P = Symbol.for("react.consumer"), ce = Symbol.for("react.context"), De = Symbol.for("react.forward_ref"), oe = Symbol.for("react.suspense"), cl = Symbol.for("react.suspense_list"), xe = Symbol.for("react.memo"), Bt = Symbol.for("react.lazy"), ua = Symbol.for("react.activity"), Mn = Symbol.iterator, Ki = {}, _n = {
        isMounted: function() {
          return !1;
        },
        enqueueForceUpdate: function(m) {
          re(m, "forceUpdate");
        },
        enqueueReplaceState: function(m) {
          re(m, "replaceState");
        },
        enqueueSetState: function(m) {
          re(m, "setState");
        }
      }, eo = Object.assign, Sf = {};
      Object.freeze(Sf), Ae.prototype.isReactComponent = {}, Ae.prototype.setState = function(m, z) {
        if (typeof m != "object" && typeof m != "function" && m != null)
          throw Error(
            "takes an object of state variables to update or a function which returns an object of state variables."
          );
        this.updater.enqueueSetState(this, m, z, "setState");
      }, Ae.prototype.forceUpdate = function(m) {
        this.updater.enqueueForceUpdate(this, m, "forceUpdate");
      };
      var ll = {
        isMounted: [
          "isMounted",
          "Instead, make sure to clean up subscriptions and pending requests in componentWillUnmount to prevent memory leaks."
        ],
        replaceState: [
          "replaceState",
          "Refactor your code to use setState instead (see https://github.com/facebook/react/issues/3236)."
        ]
      }, vl;
      for (vl in ll)
        ll.hasOwnProperty(vl) && Re(vl, ll[vl]);
      Ne.prototype = Ae.prototype, ll = st.prototype = new Ne(), ll.constructor = st, eo(ll, Ae.prototype), ll.isPureReactComponent = !0;
      var Pu = Array.isArray, cs = Symbol.for("react.client.reference"), Je = {
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
      }, Un = Object.prototype.hasOwnProperty, to = console.createTask ? console.createTask : function() {
        return null;
      };
      ll = {
        react_stack_bottom_frame: function(m) {
          return m();
        }
      };
      var Su, os, Tf = {}, ei = ll.react_stack_bottom_frame.bind(
        ll,
        ue
      )(), Dl = to(K(ue)), ja = !1, zl = /\/+/g, lo = typeof reportError == "function" ? reportError : function(m) {
        if (typeof window == "object" && typeof window.ErrorEvent == "function") {
          var z = new window.ErrorEvent("error", {
            bubbles: !0,
            cancelable: !0,
            message: typeof m == "object" && m !== null && typeof m.message == "string" ? String(m.message) : String(m),
            error: m
          });
          if (!window.dispatchEvent(z)) return;
        } else if (typeof Pt == "object" && typeof Pt.emit == "function") {
          Pt.emit("uncaughtException", m);
          return;
        }
        console.error(m);
      }, Ef = !1, ao = null, un = 0, ia = !1, Ml = !1, cn = typeof queueMicrotask == "function" ? function(m) {
        queueMicrotask(function() {
          return queueMicrotask(m);
        });
      } : Dt;
      ll = Object.freeze({
        __proto__: null,
        c: function(m) {
          return ve().useMemoCache(m);
        }
      }), F.Children = {
        map: de,
        forEach: function(m, z, te) {
          de(
            m,
            function() {
              z.apply(this, arguments);
            },
            te
          );
        },
        count: function(m) {
          var z = 0;
          return de(m, function() {
            z++;
          }), z;
        },
        toArray: function(m) {
          return de(m, function(z) {
            return z;
          }) || [];
        },
        only: function(m) {
          if (!Ke(m))
            throw Error(
              "React.Children.only expected to receive a single React element child."
            );
          return m;
        }
      }, F.Component = Ae, F.Fragment = g, F.Profiler = J, F.PureComponent = st, F.StrictMode = w, F.Suspense = oe, F.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = Je, F.__COMPILER_RUNTIME = ll, F.act = function(m) {
        var z = Je.actQueue, te = un;
        un++;
        var ne = Je.actQueue = z !== null ? z : [], pe = !1;
        try {
          var we = m();
        } catch (Ye) {
          Je.thrownErrors.push(Ye);
        }
        if (0 < Je.thrownErrors.length)
          throw le(z, te), m = Ht(Je.thrownErrors), Je.thrownErrors.length = 0, m;
        if (we !== null && typeof we == "object" && typeof we.then == "function") {
          var Ge = we;
          return cn(function() {
            pe || ia || (ia = !0, console.error(
              "You called act(async () => ...) without await. This could lead to unexpected testing behaviour, interleaving multiple act calls and mixing their scopes. You should - await act(async () => ...);"
            ));
          }), {
            then: function(Ye, al) {
              pe = !0, Ge.then(
                function(on) {
                  if (le(z, te), te === 0) {
                    try {
                      X(ne), Dt(function() {
                        return R(
                          on,
                          Ye,
                          al
                        );
                      });
                    } catch (Hh) {
                      Je.thrownErrors.push(Hh);
                    }
                    if (0 < Je.thrownErrors.length) {
                      var fs = Ht(
                        Je.thrownErrors
                      );
                      Je.thrownErrors.length = 0, al(fs);
                    }
                  } else Ye(on);
                },
                function(on) {
                  le(z, te), 0 < Je.thrownErrors.length && (on = Ht(
                    Je.thrownErrors
                  ), Je.thrownErrors.length = 0), al(on);
                }
              );
            }
          };
        }
        var ut = we;
        if (le(z, te), te === 0 && (X(ne), ne.length !== 0 && cn(function() {
          pe || ia || (ia = !0, console.error(
            "A component suspended inside an `act` scope, but the `act` call was not awaited. When testing React components that depend on asynchronous data, you must await the result:\n\nawait act(() => ...)"
          ));
        }), Je.actQueue = null), 0 < Je.thrownErrors.length)
          throw m = Ht(Je.thrownErrors), Je.thrownErrors.length = 0, m;
        return {
          then: function(Ye, al) {
            pe = !0, te === 0 ? (Je.actQueue = ne, Dt(function() {
              return R(
                ut,
                Ye,
                al
              );
            })) : Ye(ut);
          }
        };
      }, F.cache = function(m) {
        return function() {
          return m.apply(null, arguments);
        };
      }, F.captureOwnerStack = function() {
        var m = Je.getCurrentStack;
        return m === null ? null : m();
      }, F.cloneElement = function(m, z, te) {
        if (m == null)
          throw Error(
            "The argument must be a React element, but you passed " + m + "."
          );
        var ne = eo({}, m.props), pe = m.key, we = m._owner;
        if (z != null) {
          var Ge;
          e: {
            if (Un.call(z, "ref") && (Ge = Object.getOwnPropertyDescriptor(
              z,
              "ref"
            ).get) && Ge.isReactWarning) {
              Ge = !1;
              break e;
            }
            Ge = z.ref !== void 0;
          }
          Ge && (we = D()), Oe(z) && (k(z.key), pe = "" + z.key);
          for (ut in z)
            !Un.call(z, ut) || ut === "key" || ut === "__self" || ut === "__source" || ut === "ref" && z.ref === void 0 || (ne[ut] = z[ut]);
        }
        var ut = arguments.length - 2;
        if (ut === 1) ne.children = te;
        else if (1 < ut) {
          Ge = Array(ut);
          for (var Ye = 0; Ye < ut; Ye++)
            Ge[Ye] = arguments[Ye + 2];
          ne.children = Ge;
        }
        for (ne = Pe(
          m.type,
          pe,
          void 0,
          void 0,
          we,
          ne,
          m._debugStack,
          m._debugTask
        ), pe = 2; pe < arguments.length; pe++)
          we = arguments[pe], Ke(we) && we._store && (we._store.validated = 1);
        return ne;
      }, F.createContext = function(m) {
        return m = {
          $$typeof: ce,
          _currentValue: m,
          _currentValue2: m,
          _threadCount: 0,
          Provider: null,
          Consumer: null
        }, m.Provider = m, m.Consumer = {
          $$typeof: P,
          _context: m
        }, m._currentRenderer = null, m._currentRenderer2 = null, m;
      }, F.createElement = function(m, z, te) {
        for (var ne = 2; ne < arguments.length; ne++) {
          var pe = arguments[ne];
          Ke(pe) && pe._store && (pe._store.validated = 1);
        }
        if (ne = {}, pe = null, z != null)
          for (Ye in os || !("__self" in z) || "key" in z || (os = !0, console.warn(
            "Your app (or one of its dependencies) is using an outdated JSX transform. Update to the modern JSX transform for faster performance: https://react.dev/link/new-jsx-transform"
          )), Oe(z) && (k(z.key), pe = "" + z.key), z)
            Un.call(z, Ye) && Ye !== "key" && Ye !== "__self" && Ye !== "__source" && (ne[Ye] = z[Ye]);
        var we = arguments.length - 2;
        if (we === 1) ne.children = te;
        else if (1 < we) {
          for (var Ge = Array(we), ut = 0; ut < we; ut++)
            Ge[ut] = arguments[ut + 2];
          Object.freeze && Object.freeze(Ge), ne.children = Ge;
        }
        if (m && m.defaultProps)
          for (Ye in we = m.defaultProps, we)
            ne[Ye] === void 0 && (ne[Ye] = we[Ye]);
        pe && ot(
          ne,
          typeof m == "function" ? m.displayName || m.name || "Unknown" : m
        );
        var Ye = 1e4 > Je.recentlyCreatedOwnerStacks++;
        return Pe(
          m,
          pe,
          void 0,
          void 0,
          D(),
          ne,
          Ye ? Error("react-stack-top-frame") : ei,
          Ye ? to(K(m)) : Dl
        );
      }, F.createRef = function() {
        var m = { current: null };
        return Object.seal(m), m;
      }, F.forwardRef = function(m) {
        m != null && m.$$typeof === xe ? console.error(
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
        var z = { $$typeof: De, render: m }, te;
        return Object.defineProperty(z, "displayName", {
          enumerable: !1,
          configurable: !0,
          get: function() {
            return te;
          },
          set: function(ne) {
            te = ne, m.name || m.displayName || (Object.defineProperty(m, "name", { value: ne }), m.displayName = ne);
          }
        }), z;
      }, F.isValidElement = Ke, F.lazy = function(m) {
        return {
          $$typeof: Bt,
          _payload: { _status: -1, _result: m },
          _init: Ot
        };
      }, F.memo = function(m, z) {
        m == null && console.error(
          "memo: The first argument must be a component. Instead received: %s",
          m === null ? "null" : typeof m
        ), z = {
          $$typeof: xe,
          type: m,
          compare: z === void 0 ? null : z
        };
        var te;
        return Object.defineProperty(z, "displayName", {
          enumerable: !1,
          configurable: !0,
          get: function() {
            return te;
          },
          set: function(ne) {
            te = ne, m.name || m.displayName || (Object.defineProperty(m, "name", { value: ne }), m.displayName = ne);
          }
        }), z;
      }, F.startTransition = function(m) {
        var z = Je.T, te = {};
        Je.T = te, te._updatedFibers = /* @__PURE__ */ new Set();
        try {
          var ne = m(), pe = Je.S;
          pe !== null && pe(te, ne), typeof ne == "object" && ne !== null && typeof ne.then == "function" && ne.then(ze, lo);
        } catch (we) {
          lo(we);
        } finally {
          z === null && te._updatedFibers && (m = te._updatedFibers.size, te._updatedFibers.clear(), 10 < m && console.warn(
            "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
          )), Je.T = z;
        }
      }, F.unstable_useCacheRefresh = function() {
        return ve().useCacheRefresh();
      }, F.use = function(m) {
        return ve().use(m);
      }, F.useActionState = function(m, z, te) {
        return ve().useActionState(
          m,
          z,
          te
        );
      }, F.useCallback = function(m, z) {
        return ve().useCallback(m, z);
      }, F.useContext = function(m) {
        var z = ve();
        return m.$$typeof === P && console.error(
          "Calling useContext(Context.Consumer) is not supported and will cause bugs. Did you mean to call useContext(Context) instead?"
        ), z.useContext(m);
      }, F.useDebugValue = function(m, z) {
        return ve().useDebugValue(m, z);
      }, F.useDeferredValue = function(m, z) {
        return ve().useDeferredValue(m, z);
      }, F.useEffect = function(m, z, te) {
        m == null && console.warn(
          "React Hook useEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        );
        var ne = ve();
        if (typeof te == "function")
          throw Error(
            "useEffect CRUD overload is not enabled in this build of React."
          );
        return ne.useEffect(m, z);
      }, F.useId = function() {
        return ve().useId();
      }, F.useImperativeHandle = function(m, z, te) {
        return ve().useImperativeHandle(m, z, te);
      }, F.useInsertionEffect = function(m, z) {
        return m == null && console.warn(
          "React Hook useInsertionEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        ), ve().useInsertionEffect(m, z);
      }, F.useLayoutEffect = function(m, z) {
        return m == null && console.warn(
          "React Hook useLayoutEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        ), ve().useLayoutEffect(m, z);
      }, F.useMemo = function(m, z) {
        return ve().useMemo(m, z);
      }, F.useOptimistic = function(m, z) {
        return ve().useOptimistic(m, z);
      }, F.useReducer = function(m, z, te) {
        return ve().useReducer(m, z, te);
      }, F.useRef = function(m) {
        return ve().useRef(m);
      }, F.useState = function(m) {
        return ve().useState(m);
      }, F.useSyncExternalStore = function(m, z, te) {
        return ve().useSyncExternalStore(
          m,
          z,
          te
        );
      }, F.useTransition = function() {
        return ve().useTransition();
      }, F.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    }();
  }(Dp, Dp.exports)), Dp.exports;
}
var Zb;
function xh() {
  return Zb || (Zb = 1, Pt.env.NODE_ENV === "production" ? gg.exports = ST() : gg.exports = TT()), gg.exports;
}
var Kb;
function ET() {
  return Kb || (Kb = 1, Pt.env.NODE_ENV !== "production" && function() {
    function H(g) {
      if (g == null) return null;
      if (typeof g == "function")
        return g.$$typeof === Ot ? null : g.displayName || g.name || null;
      if (typeof g == "string") return g;
      switch (g) {
        case ot:
          return "Fragment";
        case Pe:
          return "Profiler";
        case He:
          return "StrictMode";
        case be:
          return "Suspense";
        case pt:
          return "SuspenseList";
        case de:
          return "Activity";
      }
      if (typeof g == "object")
        switch (typeof g.tag == "number" && console.error(
          "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
        ), g.$$typeof) {
          case Oe:
            return "Portal";
          case Ke:
            return (g.displayName || "Context") + ".Provider";
          case Ct:
            return (g._context.displayName || "Context") + ".Consumer";
          case At:
            var w = g.render;
            return g = g.displayName, g || (g = w.displayName || w.name || "", g = g !== "" ? "ForwardRef(" + g + ")" : "ForwardRef"), g;
          case je:
            return w = g.displayName || null, w !== null ? w : H(g.type) || "Memo";
          case St:
            w = g._payload, g = g._init;
            try {
              return H(g(w));
            } catch {
            }
        }
      return null;
    }
    function F(g) {
      return "" + g;
    }
    function Re(g) {
      try {
        F(g);
        var w = !1;
      } catch {
        w = !0;
      }
      if (w) {
        w = console;
        var J = w.error, P = typeof Symbol == "function" && Symbol.toStringTag && g[Symbol.toStringTag] || g.constructor.name || "Object";
        return J.call(
          w,
          "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
          P
        ), F(g);
      }
    }
    function _(g) {
      if (g === ot) return "<>";
      if (typeof g == "object" && g !== null && g.$$typeof === St)
        return "<...>";
      try {
        var w = H(g);
        return w ? "<" + w + ">" : "<...>";
      } catch {
        return "<...>";
      }
    }
    function re() {
      var g = ve.A;
      return g === null ? null : g.getOwner();
    }
    function Ae() {
      return Error("react-stack-top-frame");
    }
    function Ne(g) {
      if (ze.call(g, "key")) {
        var w = Object.getOwnPropertyDescriptor(g, "key").get;
        if (w && w.isReactWarning) return !1;
      }
      return g.key !== void 0;
    }
    function st(g, w) {
      function J() {
        le || (le = !0, console.error(
          "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
          w
        ));
      }
      J.isReactWarning = !0, Object.defineProperty(g, "key", {
        get: J,
        configurable: !0
      });
    }
    function j() {
      var g = H(this.type);
      return R[g] || (R[g] = !0, console.error(
        "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
      )), g = this.props.ref, g !== void 0 ? g : null;
    }
    function k(g, w, J, P, ce, De, oe, cl) {
      return J = De.ref, g = {
        $$typeof: ue,
        type: g,
        key: w,
        props: De,
        _owner: ce
      }, (J !== void 0 ? J : null) !== null ? Object.defineProperty(g, "ref", {
        enumerable: !1,
        get: j
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
        value: cl
      }), Object.freeze && (Object.freeze(g.props), Object.freeze(g)), g;
    }
    function ie(g, w, J, P, ce, De, oe, cl) {
      var xe = w.children;
      if (xe !== void 0)
        if (P)
          if (Dt(xe)) {
            for (P = 0; P < xe.length; P++)
              K(xe[P]);
            Object.freeze && Object.freeze(xe);
          } else
            console.error(
              "React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead."
            );
        else K(xe);
      if (ze.call(w, "key")) {
        xe = H(g);
        var Bt = Object.keys(w).filter(function(Mn) {
          return Mn !== "key";
        });
        P = 0 < Bt.length ? "{key: someKey, " + Bt.join(": ..., ") + ": ...}" : "{key: someKey}", ge[xe + P] || (Bt = 0 < Bt.length ? "{" + Bt.join(": ..., ") + ": ...}" : "{}", console.error(
          `A props object containing a "key" prop is being spread into JSX:
  let props = %s;
  <%s {...props} />
React keys must be passed directly to JSX without using spread:
  let props = %s;
  <%s key={someKey} {...props} />`,
          P,
          xe,
          Bt,
          xe
        ), ge[xe + P] = !0);
      }
      if (xe = null, J !== void 0 && (Re(J), xe = "" + J), Ne(w) && (Re(w.key), xe = "" + w.key), "key" in w) {
        J = {};
        for (var ua in w)
          ua !== "key" && (J[ua] = w[ua]);
      } else J = w;
      return xe && st(
        J,
        typeof g == "function" ? g.displayName || g.name || "Unknown" : g
      ), k(
        g,
        xe,
        De,
        ce,
        re(),
        J,
        oe,
        cl
      );
    }
    function K(g) {
      typeof g == "object" && g !== null && g.$$typeof === ue && g._store && (g._store.validated = 1);
    }
    var D = xh(), ue = Symbol.for("react.transitional.element"), Oe = Symbol.for("react.portal"), ot = Symbol.for("react.fragment"), He = Symbol.for("react.strict_mode"), Pe = Symbol.for("react.profiler"), Ct = Symbol.for("react.consumer"), Ke = Symbol.for("react.context"), At = Symbol.for("react.forward_ref"), be = Symbol.for("react.suspense"), pt = Symbol.for("react.suspense_list"), je = Symbol.for("react.memo"), St = Symbol.for("react.lazy"), de = Symbol.for("react.activity"), Ot = Symbol.for("react.client.reference"), ve = D.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, ze = Object.prototype.hasOwnProperty, Dt = Array.isArray, Ht = console.createTask ? console.createTask : function() {
      return null;
    };
    D = {
      react_stack_bottom_frame: function(g) {
        return g();
      }
    };
    var le, R = {}, X = D.react_stack_bottom_frame.bind(
      D,
      Ae
    )(), I = Ht(_(Ae)), ge = {};
    Rp.Fragment = ot, Rp.jsx = function(g, w, J, P, ce) {
      var De = 1e4 > ve.recentlyCreatedOwnerStacks++;
      return ie(
        g,
        w,
        J,
        !1,
        P,
        ce,
        De ? Error("react-stack-top-frame") : X,
        De ? Ht(_(g)) : I
      );
    }, Rp.jsxs = function(g, w, J, P, ce) {
      var De = 1e4 > ve.recentlyCreatedOwnerStacks++;
      return ie(
        g,
        w,
        J,
        !0,
        P,
        ce,
        De ? Error("react-stack-top-frame") : X,
        De ? Ht(_(g)) : I
      );
    };
  }()), Rp;
}
var Jb;
function RT() {
  return Jb || (Jb = 1, Pt.env.NODE_ENV === "production" ? vg.exports = bT() : vg.exports = ET()), vg.exports;
}
var Rt = RT(), tl = xh(), bg = { exports: {} }, Ap = {}, Sg = { exports: {} }, G0 = {};
/**
 * @license React
 * scheduler.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var kb;
function AT() {
  return kb || (kb = 1, function(H) {
    function F(R, X) {
      var I = R.length;
      R.push(X);
      e: for (; 0 < I; ) {
        var ge = I - 1 >>> 1, g = R[ge];
        if (0 < re(g, X))
          R[ge] = X, R[I] = g, I = ge;
        else break e;
      }
    }
    function Re(R) {
      return R.length === 0 ? null : R[0];
    }
    function _(R) {
      if (R.length === 0) return null;
      var X = R[0], I = R.pop();
      if (I !== X) {
        R[0] = I;
        e: for (var ge = 0, g = R.length, w = g >>> 1; ge < w; ) {
          var J = 2 * (ge + 1) - 1, P = R[J], ce = J + 1, De = R[ce];
          if (0 > re(P, I))
            ce < g && 0 > re(De, P) ? (R[ge] = De, R[ce] = I, ge = ce) : (R[ge] = P, R[J] = I, ge = J);
          else if (ce < g && 0 > re(De, I))
            R[ge] = De, R[ce] = I, ge = ce;
          else break e;
        }
      }
      return X;
    }
    function re(R, X) {
      var I = R.sortIndex - X.sortIndex;
      return I !== 0 ? I : R.id - X.id;
    }
    if (H.unstable_now = void 0, typeof performance == "object" && typeof performance.now == "function") {
      var Ae = performance;
      H.unstable_now = function() {
        return Ae.now();
      };
    } else {
      var Ne = Date, st = Ne.now();
      H.unstable_now = function() {
        return Ne.now() - st;
      };
    }
    var j = [], k = [], ie = 1, K = null, D = 3, ue = !1, Oe = !1, ot = !1, He = !1, Pe = typeof setTimeout == "function" ? setTimeout : null, Ct = typeof clearTimeout == "function" ? clearTimeout : null, Ke = typeof setImmediate < "u" ? setImmediate : null;
    function At(R) {
      for (var X = Re(k); X !== null; ) {
        if (X.callback === null) _(k);
        else if (X.startTime <= R)
          _(k), X.sortIndex = X.expirationTime, F(j, X);
        else break;
        X = Re(k);
      }
    }
    function be(R) {
      if (ot = !1, At(R), !Oe)
        if (Re(j) !== null)
          Oe = !0, pt || (pt = !0, ze());
        else {
          var X = Re(k);
          X !== null && le(be, X.startTime - R);
        }
    }
    var pt = !1, je = -1, St = 5, de = -1;
    function Ot() {
      return He ? !0 : !(H.unstable_now() - de < St);
    }
    function ve() {
      if (He = !1, pt) {
        var R = H.unstable_now();
        de = R;
        var X = !0;
        try {
          e: {
            Oe = !1, ot && (ot = !1, Ct(je), je = -1), ue = !0;
            var I = D;
            try {
              t: {
                for (At(R), K = Re(j); K !== null && !(K.expirationTime > R && Ot()); ) {
                  var ge = K.callback;
                  if (typeof ge == "function") {
                    K.callback = null, D = K.priorityLevel;
                    var g = ge(
                      K.expirationTime <= R
                    );
                    if (R = H.unstable_now(), typeof g == "function") {
                      K.callback = g, At(R), X = !0;
                      break t;
                    }
                    K === Re(j) && _(j), At(R);
                  } else _(j);
                  K = Re(j);
                }
                if (K !== null) X = !0;
                else {
                  var w = Re(k);
                  w !== null && le(
                    be,
                    w.startTime - R
                  ), X = !1;
                }
              }
              break e;
            } finally {
              K = null, D = I, ue = !1;
            }
            X = void 0;
          }
        } finally {
          X ? ze() : pt = !1;
        }
      }
    }
    var ze;
    if (typeof Ke == "function")
      ze = function() {
        Ke(ve);
      };
    else if (typeof MessageChannel < "u") {
      var Dt = new MessageChannel(), Ht = Dt.port2;
      Dt.port1.onmessage = ve, ze = function() {
        Ht.postMessage(null);
      };
    } else
      ze = function() {
        Pe(ve, 0);
      };
    function le(R, X) {
      je = Pe(function() {
        R(H.unstable_now());
      }, X);
    }
    H.unstable_IdlePriority = 5, H.unstable_ImmediatePriority = 1, H.unstable_LowPriority = 4, H.unstable_NormalPriority = 3, H.unstable_Profiling = null, H.unstable_UserBlockingPriority = 2, H.unstable_cancelCallback = function(R) {
      R.callback = null;
    }, H.unstable_forceFrameRate = function(R) {
      0 > R || 125 < R ? console.error(
        "forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"
      ) : St = 0 < R ? Math.floor(1e3 / R) : 5;
    }, H.unstable_getCurrentPriorityLevel = function() {
      return D;
    }, H.unstable_next = function(R) {
      switch (D) {
        case 1:
        case 2:
        case 3:
          var X = 3;
          break;
        default:
          X = D;
      }
      var I = D;
      D = X;
      try {
        return R();
      } finally {
        D = I;
      }
    }, H.unstable_requestPaint = function() {
      He = !0;
    }, H.unstable_runWithPriority = function(R, X) {
      switch (R) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
          break;
        default:
          R = 3;
      }
      var I = D;
      D = R;
      try {
        return X();
      } finally {
        D = I;
      }
    }, H.unstable_scheduleCallback = function(R, X, I) {
      var ge = H.unstable_now();
      switch (typeof I == "object" && I !== null ? (I = I.delay, I = typeof I == "number" && 0 < I ? ge + I : ge) : I = ge, R) {
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
      return g = I + g, R = {
        id: ie++,
        callback: X,
        priorityLevel: R,
        startTime: I,
        expirationTime: g,
        sortIndex: -1
      }, I > ge ? (R.sortIndex = I, F(k, R), Re(j) === null && R === Re(k) && (ot ? (Ct(je), je = -1) : ot = !0, le(be, I - ge))) : (R.sortIndex = g, F(j, R), Oe || ue || (Oe = !0, pt || (pt = !0, ze()))), R;
    }, H.unstable_shouldYield = Ot, H.unstable_wrapCallback = function(R) {
      var X = D;
      return function() {
        var I = D;
        D = X;
        try {
          return R.apply(this, arguments);
        } finally {
          D = I;
        }
      };
    };
  }(G0)), G0;
}
var L0 = {}, $b;
function OT() {
  return $b || ($b = 1, function(H) {
    Pt.env.NODE_ENV !== "production" && function() {
      function F() {
        if (be = !1, de) {
          var R = H.unstable_now();
          ze = R;
          var X = !0;
          try {
            e: {
              Ke = !1, At && (At = !1, je(Ot), Ot = -1), Ct = !0;
              var I = Pe;
              try {
                t: {
                  for (Ne(R), He = _(ue); He !== null && !(He.expirationTime > R && j()); ) {
                    var ge = He.callback;
                    if (typeof ge == "function") {
                      He.callback = null, Pe = He.priorityLevel;
                      var g = ge(
                        He.expirationTime <= R
                      );
                      if (R = H.unstable_now(), typeof g == "function") {
                        He.callback = g, Ne(R), X = !0;
                        break t;
                      }
                      He === _(ue) && re(ue), Ne(R);
                    } else re(ue);
                    He = _(ue);
                  }
                  if (He !== null) X = !0;
                  else {
                    var w = _(Oe);
                    w !== null && k(
                      st,
                      w.startTime - R
                    ), X = !1;
                  }
                }
                break e;
              } finally {
                He = null, Pe = I, Ct = !1;
              }
              X = void 0;
            }
          } finally {
            X ? Dt() : de = !1;
          }
        }
      }
      function Re(R, X) {
        var I = R.length;
        R.push(X);
        e: for (; 0 < I; ) {
          var ge = I - 1 >>> 1, g = R[ge];
          if (0 < Ae(g, X))
            R[ge] = X, R[I] = g, I = ge;
          else break e;
        }
      }
      function _(R) {
        return R.length === 0 ? null : R[0];
      }
      function re(R) {
        if (R.length === 0) return null;
        var X = R[0], I = R.pop();
        if (I !== X) {
          R[0] = I;
          e: for (var ge = 0, g = R.length, w = g >>> 1; ge < w; ) {
            var J = 2 * (ge + 1) - 1, P = R[J], ce = J + 1, De = R[ce];
            if (0 > Ae(P, I))
              ce < g && 0 > Ae(De, P) ? (R[ge] = De, R[ce] = I, ge = ce) : (R[ge] = P, R[J] = I, ge = J);
            else if (ce < g && 0 > Ae(De, I))
              R[ge] = De, R[ce] = I, ge = ce;
            else break e;
          }
        }
        return X;
      }
      function Ae(R, X) {
        var I = R.sortIndex - X.sortIndex;
        return I !== 0 ? I : R.id - X.id;
      }
      function Ne(R) {
        for (var X = _(Oe); X !== null; ) {
          if (X.callback === null) re(Oe);
          else if (X.startTime <= R)
            re(Oe), X.sortIndex = X.expirationTime, Re(ue, X);
          else break;
          X = _(Oe);
        }
      }
      function st(R) {
        if (At = !1, Ne(R), !Ke)
          if (_(ue) !== null)
            Ke = !0, de || (de = !0, Dt());
          else {
            var X = _(Oe);
            X !== null && k(
              st,
              X.startTime - R
            );
          }
      }
      function j() {
        return be ? !0 : !(H.unstable_now() - ze < ve);
      }
      function k(R, X) {
        Ot = pt(function() {
          R(H.unstable_now());
        }, X);
      }
      if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error()), H.unstable_now = void 0, typeof performance == "object" && typeof performance.now == "function") {
        var ie = performance;
        H.unstable_now = function() {
          return ie.now();
        };
      } else {
        var K = Date, D = K.now();
        H.unstable_now = function() {
          return K.now() - D;
        };
      }
      var ue = [], Oe = [], ot = 1, He = null, Pe = 3, Ct = !1, Ke = !1, At = !1, be = !1, pt = typeof setTimeout == "function" ? setTimeout : null, je = typeof clearTimeout == "function" ? clearTimeout : null, St = typeof setImmediate < "u" ? setImmediate : null, de = !1, Ot = -1, ve = 5, ze = -1;
      if (typeof St == "function")
        var Dt = function() {
          St(F);
        };
      else if (typeof MessageChannel < "u") {
        var Ht = new MessageChannel(), le = Ht.port2;
        Ht.port1.onmessage = F, Dt = function() {
          le.postMessage(null);
        };
      } else
        Dt = function() {
          pt(F, 0);
        };
      H.unstable_IdlePriority = 5, H.unstable_ImmediatePriority = 1, H.unstable_LowPriority = 4, H.unstable_NormalPriority = 3, H.unstable_Profiling = null, H.unstable_UserBlockingPriority = 2, H.unstable_cancelCallback = function(R) {
        R.callback = null;
      }, H.unstable_forceFrameRate = function(R) {
        0 > R || 125 < R ? console.error(
          "forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"
        ) : ve = 0 < R ? Math.floor(1e3 / R) : 5;
      }, H.unstable_getCurrentPriorityLevel = function() {
        return Pe;
      }, H.unstable_next = function(R) {
        switch (Pe) {
          case 1:
          case 2:
          case 3:
            var X = 3;
            break;
          default:
            X = Pe;
        }
        var I = Pe;
        Pe = X;
        try {
          return R();
        } finally {
          Pe = I;
        }
      }, H.unstable_requestPaint = function() {
        be = !0;
      }, H.unstable_runWithPriority = function(R, X) {
        switch (R) {
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
            break;
          default:
            R = 3;
        }
        var I = Pe;
        Pe = R;
        try {
          return X();
        } finally {
          Pe = I;
        }
      }, H.unstable_scheduleCallback = function(R, X, I) {
        var ge = H.unstable_now();
        switch (typeof I == "object" && I !== null ? (I = I.delay, I = typeof I == "number" && 0 < I ? ge + I : ge) : I = ge, R) {
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
        return g = I + g, R = {
          id: ot++,
          callback: X,
          priorityLevel: R,
          startTime: I,
          expirationTime: g,
          sortIndex: -1
        }, I > ge ? (R.sortIndex = I, Re(Oe, R), _(ue) === null && R === _(Oe) && (At ? (je(Ot), Ot = -1) : At = !0, k(st, I - ge))) : (R.sortIndex = g, Re(ue, R), Ke || Ct || (Ke = !0, de || (de = !0, Dt()))), R;
      }, H.unstable_shouldYield = j, H.unstable_wrapCallback = function(R) {
        var X = Pe;
        return function() {
          var I = Pe;
          Pe = X;
          try {
            return R.apply(this, arguments);
          } finally {
            Pe = I;
          }
        };
      }, typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    }();
  }(L0)), L0;
}
var Wb;
function cS() {
  return Wb || (Wb = 1, Pt.env.NODE_ENV === "production" ? Sg.exports = AT() : Sg.exports = OT()), Sg.exports;
}
var Tg = { exports: {} }, Ea = {};
/**
 * @license React
 * react-dom.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var Fb;
function DT() {
  if (Fb) return Ea;
  Fb = 1;
  var H = xh();
  function F(j) {
    var k = "https://react.dev/errors/" + j;
    if (1 < arguments.length) {
      k += "?args[]=" + encodeURIComponent(arguments[1]);
      for (var ie = 2; ie < arguments.length; ie++)
        k += "&args[]=" + encodeURIComponent(arguments[ie]);
    }
    return "Minified React error #" + j + "; visit " + k + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  function Re() {
  }
  var _ = {
    d: {
      f: Re,
      r: function() {
        throw Error(F(522));
      },
      D: Re,
      C: Re,
      L: Re,
      m: Re,
      X: Re,
      S: Re,
      M: Re
    },
    p: 0,
    findDOMNode: null
  }, re = Symbol.for("react.portal");
  function Ae(j, k, ie) {
    var K = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: re,
      key: K == null ? null : "" + K,
      children: j,
      containerInfo: k,
      implementation: ie
    };
  }
  var Ne = H.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  function st(j, k) {
    if (j === "font") return "";
    if (typeof k == "string")
      return k === "use-credentials" ? k : "";
  }
  return Ea.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = _, Ea.createPortal = function(j, k) {
    var ie = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!k || k.nodeType !== 1 && k.nodeType !== 9 && k.nodeType !== 11)
      throw Error(F(299));
    return Ae(j, k, null, ie);
  }, Ea.flushSync = function(j) {
    var k = Ne.T, ie = _.p;
    try {
      if (Ne.T = null, _.p = 2, j) return j();
    } finally {
      Ne.T = k, _.p = ie, _.d.f();
    }
  }, Ea.preconnect = function(j, k) {
    typeof j == "string" && (k ? (k = k.crossOrigin, k = typeof k == "string" ? k === "use-credentials" ? k : "" : void 0) : k = null, _.d.C(j, k));
  }, Ea.prefetchDNS = function(j) {
    typeof j == "string" && _.d.D(j);
  }, Ea.preinit = function(j, k) {
    if (typeof j == "string" && k && typeof k.as == "string") {
      var ie = k.as, K = st(ie, k.crossOrigin), D = typeof k.integrity == "string" ? k.integrity : void 0, ue = typeof k.fetchPriority == "string" ? k.fetchPriority : void 0;
      ie === "style" ? _.d.S(
        j,
        typeof k.precedence == "string" ? k.precedence : void 0,
        {
          crossOrigin: K,
          integrity: D,
          fetchPriority: ue
        }
      ) : ie === "script" && _.d.X(j, {
        crossOrigin: K,
        integrity: D,
        fetchPriority: ue,
        nonce: typeof k.nonce == "string" ? k.nonce : void 0
      });
    }
  }, Ea.preinitModule = function(j, k) {
    if (typeof j == "string")
      if (typeof k == "object" && k !== null) {
        if (k.as == null || k.as === "script") {
          var ie = st(
            k.as,
            k.crossOrigin
          );
          _.d.M(j, {
            crossOrigin: ie,
            integrity: typeof k.integrity == "string" ? k.integrity : void 0,
            nonce: typeof k.nonce == "string" ? k.nonce : void 0
          });
        }
      } else k == null && _.d.M(j);
  }, Ea.preload = function(j, k) {
    if (typeof j == "string" && typeof k == "object" && k !== null && typeof k.as == "string") {
      var ie = k.as, K = st(ie, k.crossOrigin);
      _.d.L(j, ie, {
        crossOrigin: K,
        integrity: typeof k.integrity == "string" ? k.integrity : void 0,
        nonce: typeof k.nonce == "string" ? k.nonce : void 0,
        type: typeof k.type == "string" ? k.type : void 0,
        fetchPriority: typeof k.fetchPriority == "string" ? k.fetchPriority : void 0,
        referrerPolicy: typeof k.referrerPolicy == "string" ? k.referrerPolicy : void 0,
        imageSrcSet: typeof k.imageSrcSet == "string" ? k.imageSrcSet : void 0,
        imageSizes: typeof k.imageSizes == "string" ? k.imageSizes : void 0,
        media: typeof k.media == "string" ? k.media : void 0
      });
    }
  }, Ea.preloadModule = function(j, k) {
    if (typeof j == "string")
      if (k) {
        var ie = st(k.as, k.crossOrigin);
        _.d.m(j, {
          as: typeof k.as == "string" && k.as !== "script" ? k.as : void 0,
          crossOrigin: ie,
          integrity: typeof k.integrity == "string" ? k.integrity : void 0
        });
      } else _.d.m(j);
  }, Ea.requestFormReset = function(j) {
    _.d.r(j);
  }, Ea.unstable_batchedUpdates = function(j, k) {
    return j(k);
  }, Ea.useFormState = function(j, k, ie) {
    return Ne.H.useFormState(j, k, ie);
  }, Ea.useFormStatus = function() {
    return Ne.H.useHostTransitionStatus();
  }, Ea.version = "19.1.1", Ea;
}
var Ra = {}, Ib;
function zT() {
  return Ib || (Ib = 1, Pt.env.NODE_ENV !== "production" && function() {
    function H() {
    }
    function F(K) {
      return "" + K;
    }
    function Re(K, D, ue) {
      var Oe = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
      try {
        F(Oe);
        var ot = !1;
      } catch {
        ot = !0;
      }
      return ot && (console.error(
        "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
        typeof Symbol == "function" && Symbol.toStringTag && Oe[Symbol.toStringTag] || Oe.constructor.name || "Object"
      ), F(Oe)), {
        $$typeof: k,
        key: Oe == null ? null : "" + Oe,
        children: K,
        containerInfo: D,
        implementation: ue
      };
    }
    function _(K, D) {
      if (K === "font") return "";
      if (typeof D == "string")
        return D === "use-credentials" ? D : "";
    }
    function re(K) {
      return K === null ? "`null`" : K === void 0 ? "`undefined`" : K === "" ? "an empty string" : 'something with type "' + typeof K + '"';
    }
    function Ae(K) {
      return K === null ? "`null`" : K === void 0 ? "`undefined`" : K === "" ? "an empty string" : typeof K == "string" ? JSON.stringify(K) : typeof K == "number" ? "`" + K + "`" : 'something with type "' + typeof K + '"';
    }
    function Ne() {
      var K = ie.H;
      return K === null && console.error(
        `Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:
1. You might have mismatching versions of React and the renderer (such as React DOM)
2. You might be breaking the Rules of Hooks
3. You might have more than one copy of React in the same app
See https://react.dev/link/invalid-hook-call for tips about how to debug and fix this problem.`
      ), K;
    }
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var st = xh(), j = {
      d: {
        f: H,
        r: function() {
          throw Error(
            "Invalid form element. requestFormReset must be passed a form that was rendered by React."
          );
        },
        D: H,
        C: H,
        L: H,
        m: H,
        X: H,
        S: H,
        M: H
      },
      p: 0,
      findDOMNode: null
    }, k = Symbol.for("react.portal"), ie = st.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    typeof Map == "function" && Map.prototype != null && typeof Map.prototype.forEach == "function" && typeof Set == "function" && Set.prototype != null && typeof Set.prototype.clear == "function" && typeof Set.prototype.forEach == "function" || console.error(
      "React depends on Map and Set built-in types. Make sure that you load a polyfill in older browsers. https://reactjs.org/link/react-polyfills"
    ), Ra.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = j, Ra.createPortal = function(K, D) {
      var ue = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
      if (!D || D.nodeType !== 1 && D.nodeType !== 9 && D.nodeType !== 11)
        throw Error("Target container is not a DOM element.");
      return Re(K, D, null, ue);
    }, Ra.flushSync = function(K) {
      var D = ie.T, ue = j.p;
      try {
        if (ie.T = null, j.p = 2, K)
          return K();
      } finally {
        ie.T = D, j.p = ue, j.d.f() && console.error(
          "flushSync was called from inside a lifecycle method. React cannot flush when React is already rendering. Consider moving this call to a scheduler task or micro task."
        );
      }
    }, Ra.preconnect = function(K, D) {
      typeof K == "string" && K ? D != null && typeof D != "object" ? console.error(
        "ReactDOM.preconnect(): Expected the `options` argument (second) to be an object but encountered %s instead. The only supported option at this time is `crossOrigin` which accepts a string.",
        Ae(D)
      ) : D != null && typeof D.crossOrigin != "string" && console.error(
        "ReactDOM.preconnect(): Expected the `crossOrigin` option (second argument) to be a string but encountered %s instead. Try removing this option or passing a string value instead.",
        re(D.crossOrigin)
      ) : console.error(
        "ReactDOM.preconnect(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
        re(K)
      ), typeof K == "string" && (D ? (D = D.crossOrigin, D = typeof D == "string" ? D === "use-credentials" ? D : "" : void 0) : D = null, j.d.C(K, D));
    }, Ra.prefetchDNS = function(K) {
      if (typeof K != "string" || !K)
        console.error(
          "ReactDOM.prefetchDNS(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
          re(K)
        );
      else if (1 < arguments.length) {
        var D = arguments[1];
        typeof D == "object" && D.hasOwnProperty("crossOrigin") ? console.error(
          "ReactDOM.prefetchDNS(): Expected only one argument, `href`, but encountered %s as a second argument instead. This argument is reserved for future options and is currently disallowed. It looks like the you are attempting to set a crossOrigin property for this DNS lookup hint. Browsers do not perform DNS queries using CORS and setting this attribute on the resource hint has no effect. Try calling ReactDOM.prefetchDNS() with just a single string argument, `href`.",
          Ae(D)
        ) : console.error(
          "ReactDOM.prefetchDNS(): Expected only one argument, `href`, but encountered %s as a second argument instead. This argument is reserved for future options and is currently disallowed. Try calling ReactDOM.prefetchDNS() with just a single string argument, `href`.",
          Ae(D)
        );
      }
      typeof K == "string" && j.d.D(K);
    }, Ra.preinit = function(K, D) {
      if (typeof K == "string" && K ? D == null || typeof D != "object" ? console.error(
        "ReactDOM.preinit(): Expected the `options` argument (second) to be an object with an `as` property describing the type of resource to be preinitialized but encountered %s instead.",
        Ae(D)
      ) : D.as !== "style" && D.as !== "script" && console.error(
        'ReactDOM.preinit(): Expected the `as` property in the `options` argument (second) to contain a valid value describing the type of resource to be preinitialized but encountered %s instead. Valid values for `as` are "style" and "script".',
        Ae(D.as)
      ) : console.error(
        "ReactDOM.preinit(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
        re(K)
      ), typeof K == "string" && D && typeof D.as == "string") {
        var ue = D.as, Oe = _(ue, D.crossOrigin), ot = typeof D.integrity == "string" ? D.integrity : void 0, He = typeof D.fetchPriority == "string" ? D.fetchPriority : void 0;
        ue === "style" ? j.d.S(
          K,
          typeof D.precedence == "string" ? D.precedence : void 0,
          {
            crossOrigin: Oe,
            integrity: ot,
            fetchPriority: He
          }
        ) : ue === "script" && j.d.X(K, {
          crossOrigin: Oe,
          integrity: ot,
          fetchPriority: He,
          nonce: typeof D.nonce == "string" ? D.nonce : void 0
        });
      }
    }, Ra.preinitModule = function(K, D) {
      var ue = "";
      if (typeof K == "string" && K || (ue += " The `href` argument encountered was " + re(K) + "."), D !== void 0 && typeof D != "object" ? ue += " The `options` argument encountered was " + re(D) + "." : D && "as" in D && D.as !== "script" && (ue += " The `as` option encountered was " + Ae(D.as) + "."), ue)
        console.error(
          "ReactDOM.preinitModule(): Expected up to two arguments, a non-empty `href` string and, optionally, an `options` object with a valid `as` property.%s",
          ue
        );
      else
        switch (ue = D && typeof D.as == "string" ? D.as : "script", ue) {
          case "script":
            break;
          default:
            ue = Ae(ue), console.error(
              'ReactDOM.preinitModule(): Currently the only supported "as" type for this function is "script" but received "%s" instead. This warning was generated for `href` "%s". In the future other module types will be supported, aligning with the import-attributes proposal. Learn more here: (https://github.com/tc39/proposal-import-attributes)',
              ue,
              K
            );
        }
      typeof K == "string" && (typeof D == "object" && D !== null ? (D.as == null || D.as === "script") && (ue = _(
        D.as,
        D.crossOrigin
      ), j.d.M(K, {
        crossOrigin: ue,
        integrity: typeof D.integrity == "string" ? D.integrity : void 0,
        nonce: typeof D.nonce == "string" ? D.nonce : void 0
      })) : D == null && j.d.M(K));
    }, Ra.preload = function(K, D) {
      var ue = "";
      if (typeof K == "string" && K || (ue += " The `href` argument encountered was " + re(K) + "."), D == null || typeof D != "object" ? ue += " The `options` argument encountered was " + re(D) + "." : typeof D.as == "string" && D.as || (ue += " The `as` option encountered was " + re(D.as) + "."), ue && console.error(
        'ReactDOM.preload(): Expected two arguments, a non-empty `href` string and an `options` object with an `as` property valid for a `<link rel="preload" as="..." />` tag.%s',
        ue
      ), typeof K == "string" && typeof D == "object" && D !== null && typeof D.as == "string") {
        ue = D.as;
        var Oe = _(
          ue,
          D.crossOrigin
        );
        j.d.L(K, ue, {
          crossOrigin: Oe,
          integrity: typeof D.integrity == "string" ? D.integrity : void 0,
          nonce: typeof D.nonce == "string" ? D.nonce : void 0,
          type: typeof D.type == "string" ? D.type : void 0,
          fetchPriority: typeof D.fetchPriority == "string" ? D.fetchPriority : void 0,
          referrerPolicy: typeof D.referrerPolicy == "string" ? D.referrerPolicy : void 0,
          imageSrcSet: typeof D.imageSrcSet == "string" ? D.imageSrcSet : void 0,
          imageSizes: typeof D.imageSizes == "string" ? D.imageSizes : void 0,
          media: typeof D.media == "string" ? D.media : void 0
        });
      }
    }, Ra.preloadModule = function(K, D) {
      var ue = "";
      typeof K == "string" && K || (ue += " The `href` argument encountered was " + re(K) + "."), D !== void 0 && typeof D != "object" ? ue += " The `options` argument encountered was " + re(D) + "." : D && "as" in D && typeof D.as != "string" && (ue += " The `as` option encountered was " + re(D.as) + "."), ue && console.error(
        'ReactDOM.preloadModule(): Expected two arguments, a non-empty `href` string and, optionally, an `options` object with an `as` property valid for a `<link rel="modulepreload" as="..." />` tag.%s',
        ue
      ), typeof K == "string" && (D ? (ue = _(
        D.as,
        D.crossOrigin
      ), j.d.m(K, {
        as: typeof D.as == "string" && D.as !== "script" ? D.as : void 0,
        crossOrigin: ue,
        integrity: typeof D.integrity == "string" ? D.integrity : void 0
      })) : j.d.m(K));
    }, Ra.requestFormReset = function(K) {
      j.d.r(K);
    }, Ra.unstable_batchedUpdates = function(K, D) {
      return K(D);
    }, Ra.useFormState = function(K, D, ue) {
      return Ne().useFormState(K, D, ue);
    }, Ra.useFormStatus = function() {
      return Ne().useHostTransitionStatus();
    }, Ra.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  }()), Ra;
}
var Pb;
function oS() {
  if (Pb) return Tg.exports;
  Pb = 1;
  function H() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) {
      if (Pt.env.NODE_ENV !== "production")
        throw new Error("^_^");
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(H);
      } catch (F) {
        console.error(F);
      }
    }
  }
  return Pt.env.NODE_ENV === "production" ? (H(), Tg.exports = DT()) : Tg.exports = zT(), Tg.exports;
}
var eS;
function MT() {
  if (eS) return Ap;
  eS = 1;
  var H = cS(), F = xh(), Re = oS();
  function _(l) {
    var n = "https://react.dev/errors/" + l;
    if (1 < arguments.length) {
      n += "?args[]=" + encodeURIComponent(arguments[1]);
      for (var u = 2; u < arguments.length; u++)
        n += "&args[]=" + encodeURIComponent(arguments[u]);
    }
    return "Minified React error #" + l + "; visit " + n + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  function re(l) {
    return !(!l || l.nodeType !== 1 && l.nodeType !== 9 && l.nodeType !== 11);
  }
  function Ae(l) {
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
  function Ne(l) {
    if (l.tag === 13) {
      var n = l.memoizedState;
      if (n === null && (l = l.alternate, l !== null && (n = l.memoizedState)), n !== null) return n.dehydrated;
    }
    return null;
  }
  function st(l) {
    if (Ae(l) !== l)
      throw Error(_(188));
  }
  function j(l) {
    var n = l.alternate;
    if (!n) {
      if (n = Ae(l), n === null) throw Error(_(188));
      return n !== l ? null : l;
    }
    for (var u = l, c = n; ; ) {
      var r = u.return;
      if (r === null) break;
      var s = r.alternate;
      if (s === null) {
        if (c = r.return, c !== null) {
          u = c;
          continue;
        }
        break;
      }
      if (r.child === s.child) {
        for (s = r.child; s; ) {
          if (s === u) return st(r), l;
          if (s === c) return st(r), n;
          s = s.sibling;
        }
        throw Error(_(188));
      }
      if (u.return !== c.return) u = r, c = s;
      else {
        for (var y = !1, p = r.child; p; ) {
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
        if (!y) {
          for (p = s.child; p; ) {
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
          if (!y) throw Error(_(189));
        }
      }
      if (u.alternate !== c) throw Error(_(190));
    }
    if (u.tag !== 3) throw Error(_(188));
    return u.stateNode.current === u ? l : n;
  }
  function k(l) {
    var n = l.tag;
    if (n === 5 || n === 26 || n === 27 || n === 6) return l;
    for (l = l.child; l !== null; ) {
      if (n = k(l), n !== null) return n;
      l = l.sibling;
    }
    return null;
  }
  var ie = Object.assign, K = Symbol.for("react.element"), D = Symbol.for("react.transitional.element"), ue = Symbol.for("react.portal"), Oe = Symbol.for("react.fragment"), ot = Symbol.for("react.strict_mode"), He = Symbol.for("react.profiler"), Pe = Symbol.for("react.provider"), Ct = Symbol.for("react.consumer"), Ke = Symbol.for("react.context"), At = Symbol.for("react.forward_ref"), be = Symbol.for("react.suspense"), pt = Symbol.for("react.suspense_list"), je = Symbol.for("react.memo"), St = Symbol.for("react.lazy"), de = Symbol.for("react.activity"), Ot = Symbol.for("react.memo_cache_sentinel"), ve = Symbol.iterator;
  function ze(l) {
    return l === null || typeof l != "object" ? null : (l = ve && l[ve] || l["@@iterator"], typeof l == "function" ? l : null);
  }
  var Dt = Symbol.for("react.client.reference");
  function Ht(l) {
    if (l == null) return null;
    if (typeof l == "function")
      return l.$$typeof === Dt ? null : l.displayName || l.name || null;
    if (typeof l == "string") return l;
    switch (l) {
      case Oe:
        return "Fragment";
      case He:
        return "Profiler";
      case ot:
        return "StrictMode";
      case be:
        return "Suspense";
      case pt:
        return "SuspenseList";
      case de:
        return "Activity";
    }
    if (typeof l == "object")
      switch (l.$$typeof) {
        case ue:
          return "Portal";
        case Ke:
          return (l.displayName || "Context") + ".Provider";
        case Ct:
          return (l._context.displayName || "Context") + ".Consumer";
        case At:
          var n = l.render;
          return l = l.displayName, l || (l = n.displayName || n.name || "", l = l !== "" ? "ForwardRef(" + l + ")" : "ForwardRef"), l;
        case je:
          return n = l.displayName || null, n !== null ? n : Ht(l.type) || "Memo";
        case St:
          n = l._payload, l = l._init;
          try {
            return Ht(l(n));
          } catch {
          }
      }
    return null;
  }
  var le = Array.isArray, R = F.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, X = Re.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, I = {
    pending: !1,
    data: null,
    method: null,
    action: null
  }, ge = [], g = -1;
  function w(l) {
    return { current: l };
  }
  function J(l) {
    0 > g || (l.current = ge[g], ge[g] = null, g--);
  }
  function P(l, n) {
    g++, ge[g] = l.current, l.current = n;
  }
  var ce = w(null), De = w(null), oe = w(null), cl = w(null);
  function xe(l, n) {
    switch (P(oe, n), P(De, l), P(ce, null), n.nodeType) {
      case 9:
      case 11:
        l = (l = n.documentElement) && (l = l.namespaceURI) ? Vu(l) : 0;
        break;
      default:
        if (l = n.tagName, n = n.namespaceURI)
          n = Vu(n), l = $o(n, l);
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
    J(ce), P(ce, l);
  }
  function Bt() {
    J(ce), J(De), J(oe);
  }
  function ua(l) {
    l.memoizedState !== null && P(cl, l);
    var n = ce.current, u = $o(n, l.type);
    n !== u && (P(De, l), P(ce, u));
  }
  function Mn(l) {
    De.current === l && (J(ce), J(De)), cl.current === l && (J(cl), Sa._currentValue = I);
  }
  var Ki = Object.prototype.hasOwnProperty, _n = H.unstable_scheduleCallback, eo = H.unstable_cancelCallback, Sf = H.unstable_shouldYield, ll = H.unstable_requestPaint, vl = H.unstable_now, Pu = H.unstable_getCurrentPriorityLevel, cs = H.unstable_ImmediatePriority, Je = H.unstable_UserBlockingPriority, Un = H.unstable_NormalPriority, to = H.unstable_LowPriority, Su = H.unstable_IdlePriority, os = H.log, Tf = H.unstable_setDisableYieldValue, ei = null, Dl = null;
  function ja(l) {
    if (typeof os == "function" && Tf(l), Dl && typeof Dl.setStrictMode == "function")
      try {
        Dl.setStrictMode(ei, l);
      } catch {
      }
  }
  var zl = Math.clz32 ? Math.clz32 : ao, lo = Math.log, Ef = Math.LN2;
  function ao(l) {
    return l >>>= 0, l === 0 ? 32 : 31 - (lo(l) / Ef | 0) | 0;
  }
  var un = 256, ia = 4194304;
  function Ml(l) {
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
  function cn(l, n, u) {
    var c = l.pendingLanes;
    if (c === 0) return 0;
    var r = 0, s = l.suspendedLanes, y = l.pingedLanes;
    l = l.warmLanes;
    var p = c & 134217727;
    return p !== 0 ? (c = p & ~s, c !== 0 ? r = Ml(c) : (y &= p, y !== 0 ? r = Ml(y) : u || (u = p & ~l, u !== 0 && (r = Ml(u))))) : (p = c & ~s, p !== 0 ? r = Ml(p) : y !== 0 ? r = Ml(y) : u || (u = c & ~l, u !== 0 && (r = Ml(u)))), r === 0 ? 0 : n !== 0 && n !== r && (n & s) === 0 && (s = r & -r, u = n & -n, s >= u || s === 32 && (u & 4194048) !== 0) ? n : r;
  }
  function m(l, n) {
    return (l.pendingLanes & ~(l.suspendedLanes & ~l.pingedLanes) & n) === 0;
  }
  function z(l, n) {
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
  function te() {
    var l = un;
    return un <<= 1, (un & 4194048) === 0 && (un = 256), l;
  }
  function ne() {
    var l = ia;
    return ia <<= 1, (ia & 62914560) === 0 && (ia = 4194304), l;
  }
  function pe(l) {
    for (var n = [], u = 0; 31 > u; u++) n.push(l);
    return n;
  }
  function we(l, n) {
    l.pendingLanes |= n, n !== 268435456 && (l.suspendedLanes = 0, l.pingedLanes = 0, l.warmLanes = 0);
  }
  function Ge(l, n, u, c, r, s) {
    var y = l.pendingLanes;
    l.pendingLanes = u, l.suspendedLanes = 0, l.pingedLanes = 0, l.warmLanes = 0, l.expiredLanes &= u, l.entangledLanes &= u, l.errorRecoveryDisabledLanes &= u, l.shellSuspendCounter = 0;
    var p = l.entanglements, S = l.expirationTimes, C = l.hiddenUpdates;
    for (u = y & ~u; 0 < u; ) {
      var Z = 31 - zl(u), W = 1 << Z;
      p[Z] = 0, S[Z] = -1;
      var N = C[Z];
      if (N !== null)
        for (C[Z] = null, Z = 0; Z < N.length; Z++) {
          var B = N[Z];
          B !== null && (B.lane &= -536870913);
        }
      u &= ~W;
    }
    c !== 0 && ut(l, c, 0), s !== 0 && r === 0 && l.tag !== 0 && (l.suspendedLanes |= s & ~(y & ~n));
  }
  function ut(l, n, u) {
    l.pendingLanes |= n, l.suspendedLanes &= ~n;
    var c = 31 - zl(n);
    l.entangledLanes |= n, l.entanglements[c] = l.entanglements[c] | 1073741824 | u & 4194090;
  }
  function Ye(l, n) {
    var u = l.entangledLanes |= n;
    for (l = l.entanglements; u; ) {
      var c = 31 - zl(u), r = 1 << c;
      r & n | l[c] & n && (l[c] |= n), u &= ~r;
    }
  }
  function al(l) {
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
  function on(l) {
    return l &= -l, 2 < l ? 8 < l ? (l & 134217727) !== 0 ? 32 : 268435456 : 8 : 2;
  }
  function fs() {
    var l = X.p;
    return l !== 0 ? l : (l = window.event, l === void 0 ? 32 : Bm(l.type));
  }
  function Hh(l, n) {
    var u = X.p;
    try {
      return X.p = l, n();
    } finally {
      X.p = u;
    }
  }
  var ol = Math.random().toString(36).slice(2), gl = "__reactFiber$" + ol, $l = "__reactProps$" + ol, no = "__reactContainer$" + ol, rs = "__reactEvents$" + ol, zp = "__reactListeners$" + ol, ss = "__reactHandles$" + ol, Mp = "__reactResources$" + ol, he = "__reactMarker$" + ol;
  function Rf(l) {
    delete l[gl], delete l[$l], delete l[rs], delete l[zp], delete l[ss];
  }
  function _l(l) {
    var n = l[gl];
    if (n) return n;
    for (var u = l.parentNode; u; ) {
      if (n = u[no] || u[gl]) {
        if (u = n.alternate, n.child !== null || u !== null && u.child !== null)
          for (l = ql(l); l !== null; ) {
            if (u = l[gl]) return u;
            l = ql(l);
          }
        return n;
      }
      l = u, u = l.parentNode;
    }
    return null;
  }
  function Ji(l) {
    if (l = l[gl] || l[no]) {
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
  function Tu(l) {
    var n = l[Mp];
    return n || (n = l[Mp] = { hoistableStyles: /* @__PURE__ */ new Map(), hoistableScripts: /* @__PURE__ */ new Map() }), n;
  }
  function fl(l) {
    l[he] = !0;
  }
  var Of = /* @__PURE__ */ new Set(), Aa = {};
  function ti(l, n) {
    li(l, n), li(l + "Capture", n);
  }
  function li(l, n) {
    for (Aa[l] = n, l = 0; l < n.length; l++)
      Of.add(n[l]);
  }
  var _p = RegExp(
    "^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
  ), ds = {}, Nh = {};
  function Up(l) {
    return Ki.call(Nh, l) ? !0 : Ki.call(ds, l) ? !1 : _p.test(l) ? Nh[l] = !0 : (ds[l] = !0, !1);
  }
  function Eu(l, n, u) {
    if (Up(n))
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
  function Df(l, n, u) {
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
  function Cn(l, n, u, c) {
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
  var hs, wh;
  function ki(l) {
    if (hs === void 0)
      try {
        throw Error();
      } catch (u) {
        var n = u.stack.trim().match(/\n( *(at )?)/);
        hs = n && n[1] || "", wh = -1 < u.stack.indexOf(`
    at`) ? " (<anonymous>)" : -1 < u.stack.indexOf("@") ? "@unknown:0:0" : "";
      }
    return `
` + hs + l + wh;
  }
  var Wl = !1;
  function ai(l, n) {
    if (!l || Wl) return "";
    Wl = !0;
    var u = Error.prepareStackTrace;
    Error.prepareStackTrace = void 0;
    try {
      var c = {
        DetermineComponentFrameRoot: function() {
          try {
            if (n) {
              var W = function() {
                throw Error();
              };
              if (Object.defineProperty(W.prototype, "props", {
                set: function() {
                  throw Error();
                }
              }), typeof Reflect == "object" && Reflect.construct) {
                try {
                  Reflect.construct(W, []);
                } catch (B) {
                  var N = B;
                }
                Reflect.construct(l, [], W);
              } else {
                try {
                  W.call();
                } catch (B) {
                  N = B;
                }
                l.call(W.prototype);
              }
            } else {
              try {
                throw Error();
              } catch (B) {
                N = B;
              }
              (W = l()) && typeof W.catch == "function" && W.catch(function() {
              });
            }
          } catch (B) {
            if (B && N && typeof B.stack == "string")
              return [B.stack, N.stack];
          }
          return [null, null];
        }
      };
      c.DetermineComponentFrameRoot.displayName = "DetermineComponentFrameRoot";
      var r = Object.getOwnPropertyDescriptor(
        c.DetermineComponentFrameRoot,
        "name"
      );
      r && r.configurable && Object.defineProperty(
        c.DetermineComponentFrameRoot,
        "name",
        { value: "DetermineComponentFrameRoot" }
      );
      var s = c.DetermineComponentFrameRoot(), y = s[0], p = s[1];
      if (y && p) {
        var S = y.split(`
`), C = p.split(`
`);
        for (r = c = 0; c < S.length && !S[c].includes("DetermineComponentFrameRoot"); )
          c++;
        for (; r < C.length && !C[r].includes(
          "DetermineComponentFrameRoot"
        ); )
          r++;
        if (c === S.length || r === C.length)
          for (c = S.length - 1, r = C.length - 1; 1 <= c && 0 <= r && S[c] !== C[r]; )
            r--;
        for (; 1 <= c && 0 <= r; c--, r--)
          if (S[c] !== C[r]) {
            if (c !== 1 || r !== 1)
              do
                if (c--, r--, 0 > r || S[c] !== C[r]) {
                  var Z = `
` + S[c].replace(" at new ", " at ");
                  return l.displayName && Z.includes("<anonymous>") && (Z = Z.replace("<anonymous>", l.displayName)), Z;
                }
              while (1 <= c && 0 <= r);
            break;
          }
      }
    } finally {
      Wl = !1, Error.prepareStackTrace = u;
    }
    return (u = l ? l.displayName || l.name : "") ? ki(u) : "";
  }
  function $i(l) {
    switch (l.tag) {
      case 26:
      case 27:
      case 5:
        return ki(l.type);
      case 16:
        return ki("Lazy");
      case 13:
        return ki("Suspense");
      case 19:
        return ki("SuspenseList");
      case 0:
      case 15:
        return ai(l.type, !1);
      case 11:
        return ai(l.type.render, !1);
      case 1:
        return ai(l.type, !0);
      case 31:
        return ki("Activity");
      default:
        return "";
    }
  }
  function qh(l) {
    try {
      var n = "";
      do
        n += $i(l), l = l.return;
      while (l);
      return n;
    } catch (u) {
      return `
Error generating stack: ` + u.message + `
` + u.stack;
    }
  }
  function Ll(l) {
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
  function zf(l) {
    var n = l.type;
    return (l = l.nodeName) && l.toLowerCase() === "input" && (n === "checkbox" || n === "radio");
  }
  function Bh(l) {
    var n = zf(l) ? "checked" : "value", u = Object.getOwnPropertyDescriptor(
      l.constructor.prototype,
      n
    ), c = "" + l[n];
    if (!l.hasOwnProperty(n) && typeof u < "u" && typeof u.get == "function" && typeof u.set == "function") {
      var r = u.get, s = u.set;
      return Object.defineProperty(l, n, {
        configurable: !0,
        get: function() {
          return r.call(this);
        },
        set: function(y) {
          c = "" + y, s.call(this, y);
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
  function ni(l) {
    l._valueTracker || (l._valueTracker = Bh(l));
  }
  function Wi(l) {
    if (!l) return !1;
    var n = l._valueTracker;
    if (!n) return !0;
    var u = n.getValue(), c = "";
    return l && (c = zf(l) ? l.checked ? "true" : "false" : l.value), l = c, l !== u ? (n.setValue(l), !0) : !1;
  }
  function uo(l) {
    if (l = l || (typeof document < "u" ? document : void 0), typeof l > "u") return null;
    try {
      return l.activeElement || l.body;
    } catch {
      return l.body;
    }
  }
  var Rg = /[\n"\\]/g;
  function Ga(l) {
    return l.replace(
      Rg,
      function(n) {
        return "\\" + n.charCodeAt(0).toString(16) + " ";
      }
    );
  }
  function ys(l, n, u, c, r, s, y, p) {
    l.name = "", y != null && typeof y != "function" && typeof y != "symbol" && typeof y != "boolean" ? l.type = y : l.removeAttribute("type"), n != null ? y === "number" ? (n === 0 && l.value === "" || l.value != n) && (l.value = "" + Ll(n)) : l.value !== "" + Ll(n) && (l.value = "" + Ll(n)) : y !== "submit" && y !== "reset" || l.removeAttribute("value"), n != null ? Mf(l, y, Ll(n)) : u != null ? Mf(l, y, Ll(u)) : c != null && l.removeAttribute("value"), r == null && s != null && (l.defaultChecked = !!s), r != null && (l.checked = r && typeof r != "function" && typeof r != "symbol"), p != null && typeof p != "function" && typeof p != "symbol" && typeof p != "boolean" ? l.name = "" + Ll(p) : l.removeAttribute("name");
  }
  function ms(l, n, u, c, r, s, y, p) {
    if (s != null && typeof s != "function" && typeof s != "symbol" && typeof s != "boolean" && (l.type = s), n != null || u != null) {
      if (!(s !== "submit" && s !== "reset" || n != null))
        return;
      u = u != null ? "" + Ll(u) : "", n = n != null ? "" + Ll(n) : u, p || n === l.value || (l.value = n), l.defaultValue = n;
    }
    c = c ?? r, c = typeof c != "function" && typeof c != "symbol" && !!c, l.checked = p ? l.checked : !!c, l.defaultChecked = !!c, y != null && typeof y != "function" && typeof y != "symbol" && typeof y != "boolean" && (l.name = y);
  }
  function Mf(l, n, u) {
    n === "number" && uo(l.ownerDocument) === l || l.defaultValue === "" + u || (l.defaultValue = "" + u);
  }
  function Fi(l, n, u, c) {
    if (l = l.options, n) {
      n = {};
      for (var r = 0; r < u.length; r++)
        n["$" + u[r]] = !0;
      for (u = 0; u < l.length; u++)
        r = n.hasOwnProperty("$" + l[u].value), l[u].selected !== r && (l[u].selected = r), r && c && (l[u].defaultSelected = !0);
    } else {
      for (u = "" + Ll(u), n = null, r = 0; r < l.length; r++) {
        if (l[r].value === u) {
          l[r].selected = !0, c && (l[r].defaultSelected = !0);
          return;
        }
        n !== null || l[r].disabled || (n = l[r]);
      }
      n !== null && (n.selected = !0);
    }
  }
  function Yh(l, n, u) {
    if (n != null && (n = "" + Ll(n), n !== l.value && (l.value = n), u == null)) {
      l.defaultValue !== n && (l.defaultValue = n);
      return;
    }
    l.defaultValue = u != null ? "" + Ll(u) : "";
  }
  function jh(l, n, u, c) {
    if (n == null) {
      if (c != null) {
        if (u != null) throw Error(_(92));
        if (le(c)) {
          if (1 < c.length) throw Error(_(93));
          c = c[0];
        }
        u = c;
      }
      u == null && (u = ""), n = u;
    }
    u = Ll(n), l.defaultValue = u, c = l.textContent, c === u && c !== "" && c !== null && (l.value = c);
  }
  function io(l, n) {
    if (n) {
      var u = l.firstChild;
      if (u && u === l.lastChild && u.nodeType === 3) {
        u.nodeValue = n;
        return;
      }
    }
    l.textContent = n;
  }
  var Cp = new Set(
    "animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(
      " "
    )
  );
  function ps(l, n, u) {
    var c = n.indexOf("--") === 0;
    u == null || typeof u == "boolean" || u === "" ? c ? l.setProperty(n, "") : n === "float" ? l.cssFloat = "" : l[n] = "" : c ? l.setProperty(n, u) : typeof u != "number" || u === 0 || Cp.has(n) ? n === "float" ? l.cssFloat = u : l[n] = ("" + u).trim() : l[n] = u + "px";
  }
  function _f(l, n, u) {
    if (n != null && typeof n != "object")
      throw Error(_(62));
    if (l = l.style, u != null) {
      for (var c in u)
        !u.hasOwnProperty(c) || n != null && n.hasOwnProperty(c) || (c.indexOf("--") === 0 ? l.setProperty(c, "") : c === "float" ? l.cssFloat = "" : l[c] = "");
      for (var r in n)
        c = n[r], n.hasOwnProperty(r) && u[r] !== c && ps(l, r, c);
    } else
      for (var s in n)
        n.hasOwnProperty(s) && ps(l, s, n[s]);
  }
  function Ii(l) {
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
  ]), xp = /^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;
  function Uf(l) {
    return xp.test("" + l) ? "javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')" : l;
  }
  var Pi = null;
  function vs(l) {
    return l = l.target || l.srcElement || window, l.correspondingUseElement && (l = l.correspondingUseElement), l.nodeType === 3 ? l.parentNode : l;
  }
  var co = null, oo = null;
  function Hp(l) {
    var n = Ji(l);
    if (n && (l = n.stateNode)) {
      var u = l[$l] || null;
      e: switch (l = n.stateNode, n.type) {
        case "input":
          if (ys(
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
              'input[name="' + Ga(
                "" + n
              ) + '"][type="radio"]'
            ), n = 0; n < u.length; n++) {
              var c = u[n];
              if (c !== l && c.form === l.form) {
                var r = c[$l] || null;
                if (!r) throw Error(_(90));
                ys(
                  c,
                  r.value,
                  r.defaultValue,
                  r.defaultValue,
                  r.checked,
                  r.defaultChecked,
                  r.type,
                  r.name
                );
              }
            }
            for (n = 0; n < u.length; n++)
              c = u[n], c.form === l.form && Wi(c);
          }
          break e;
        case "textarea":
          Yh(l, u.value, u.defaultValue);
          break e;
        case "select":
          n = u.value, n != null && Fi(l, !!u.multiple, n, !1);
      }
    }
  }
  var Gh = !1;
  function fo(l, n, u) {
    if (Gh) return l(n, u);
    Gh = !0;
    try {
      var c = l(n);
      return c;
    } finally {
      if (Gh = !1, (co !== null || oo !== null) && (Cc(), co && (n = co, l = oo, oo = co = null, Hp(n), l)))
        for (n = 0; n < l.length; n++) Hp(l[n]);
    }
  }
  function ec(l, n) {
    var u = l.stateNode;
    if (u === null) return null;
    var c = u[$l] || null;
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
  var xn = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), gs = !1;
  if (xn)
    try {
      var Ru = {};
      Object.defineProperty(Ru, "passive", {
        get: function() {
          gs = !0;
        }
      }), window.addEventListener("test", Ru, Ru), window.removeEventListener("test", Ru, Ru);
    } catch {
      gs = !1;
    }
  var Au = null, ro = null, tc = null;
  function Lh() {
    if (tc) return tc;
    var l, n = ro, u = n.length, c, r = "value" in Au ? Au.value : Au.textContent, s = r.length;
    for (l = 0; l < u && n[l] === r[l]; l++) ;
    var y = u - l;
    for (c = 1; c <= y && n[u - c] === r[s - c]; c++) ;
    return tc = r.slice(l, 1 < c ? 1 - c : void 0);
  }
  function Ul(l) {
    var n = l.keyCode;
    return "charCode" in l ? (l = l.charCode, l === 0 && n === 13 && (l = 13)) : l = n, l === 10 && (l = 13), 32 <= l || l === 13 ? l : 0;
  }
  function bs() {
    return !0;
  }
  function Ss() {
    return !1;
  }
  function Fl(l) {
    function n(u, c, r, s, y) {
      this._reactName = u, this._targetInst = r, this.type = c, this.nativeEvent = s, this.target = y, this.currentTarget = null;
      for (var p in l)
        l.hasOwnProperty(p) && (u = l[p], this[p] = u ? u(s) : s[p]);
      return this.isDefaultPrevented = (s.defaultPrevented != null ? s.defaultPrevented : s.returnValue === !1) ? bs : Ss, this.isPropagationStopped = Ss, this;
    }
    return ie(n.prototype, {
      preventDefault: function() {
        this.defaultPrevented = !0;
        var u = this.nativeEvent;
        u && (u.preventDefault ? u.preventDefault() : typeof u.returnValue != "unknown" && (u.returnValue = !1), this.isDefaultPrevented = bs);
      },
      stopPropagation: function() {
        var u = this.nativeEvent;
        u && (u.stopPropagation ? u.stopPropagation() : typeof u.cancelBubble != "unknown" && (u.cancelBubble = !0), this.isPropagationStopped = bs);
      },
      persist: function() {
      },
      isPersistent: bs
    }), n;
  }
  var ui = {
    eventPhase: 0,
    bubbles: 0,
    cancelable: 0,
    timeStamp: function(l) {
      return l.timeStamp || Date.now();
    },
    defaultPrevented: 0,
    isTrusted: 0
  }, Ts = Fl(ui), Cf = ie({}, ui, { view: 0, detail: 0 }), Np = Fl(Cf), Vh, Es, xf, lc = ie({}, Cf, {
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
    getModifierState: Ou,
    button: 0,
    buttons: 0,
    relatedTarget: function(l) {
      return l.relatedTarget === void 0 ? l.fromElement === l.srcElement ? l.toElement : l.fromElement : l.relatedTarget;
    },
    movementX: function(l) {
      return "movementX" in l ? l.movementX : (l !== xf && (xf && l.type === "mousemove" ? (Vh = l.screenX - xf.screenX, Es = l.screenY - xf.screenY) : Es = Vh = 0, xf = l), Vh);
    },
    movementY: function(l) {
      return "movementY" in l ? l.movementY : Es;
    }
  }), Xh = Fl(lc), wp = ie({}, lc, { dataTransfer: 0 }), qp = Fl(wp), Og = ie({}, Cf, { relatedTarget: 0 }), Qh = Fl(Og), Dg = ie({}, ui, {
    animationName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), zg = Fl(Dg), Mg = ie({}, ui, {
    clipboardData: function(l) {
      return "clipboardData" in l ? l.clipboardData : window.clipboardData;
    }
  }), Hf = Fl(Mg), Bp = ie({}, ui, { data: 0 }), Zh = Fl(Bp), Yp = {
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
  }, jp = {
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
  }, Kh = {
    Alt: "altKey",
    Control: "ctrlKey",
    Meta: "metaKey",
    Shift: "shiftKey"
  };
  function Gp(l) {
    var n = this.nativeEvent;
    return n.getModifierState ? n.getModifierState(l) : (l = Kh[l]) ? !!n[l] : !1;
  }
  function Ou() {
    return Gp;
  }
  var ac = ie({}, Cf, {
    key: function(l) {
      if (l.key) {
        var n = Yp[l.key] || l.key;
        if (n !== "Unidentified") return n;
      }
      return l.type === "keypress" ? (l = Ul(l), l === 13 ? "Enter" : String.fromCharCode(l)) : l.type === "keydown" || l.type === "keyup" ? jp[l.keyCode] || "Unidentified" : "";
    },
    code: 0,
    location: 0,
    ctrlKey: 0,
    shiftKey: 0,
    altKey: 0,
    metaKey: 0,
    repeat: 0,
    locale: 0,
    getModifierState: Ou,
    charCode: function(l) {
      return l.type === "keypress" ? Ul(l) : 0;
    },
    keyCode: function(l) {
      return l.type === "keydown" || l.type === "keyup" ? l.keyCode : 0;
    },
    which: function(l) {
      return l.type === "keypress" ? Ul(l) : l.type === "keydown" || l.type === "keyup" ? l.keyCode : 0;
    }
  }), fn = Fl(ac), Oa = ie({}, lc, {
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
  }), Nf = Fl(Oa), Rs = ie({}, Cf, {
    touches: 0,
    targetTouches: 0,
    changedTouches: 0,
    altKey: 0,
    metaKey: 0,
    ctrlKey: 0,
    shiftKey: 0,
    getModifierState: Ou
  }), Jh = Fl(Rs), ca = ie({}, ui, {
    propertyName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), Lp = Fl(ca), As = ie({}, lc, {
    deltaX: function(l) {
      return "deltaX" in l ? l.deltaX : "wheelDeltaX" in l ? -l.wheelDeltaX : 0;
    },
    deltaY: function(l) {
      return "deltaY" in l ? l.deltaY : "wheelDeltaY" in l ? -l.wheelDeltaY : "wheelDelta" in l ? -l.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), nc = Fl(As), kh = ie({}, ui, {
    newState: 0,
    oldState: 0
  }), Vp = Fl(kh), Xp = [9, 13, 27, 32], wf = xn && "CompositionEvent" in window, qf = null;
  xn && "documentMode" in document && (qf = document.documentMode);
  var $h = xn && "TextEvent" in window && !qf, Hn = xn && (!wf || qf && 8 < qf && 11 >= qf), Wh = " ", Os = !1;
  function Bf(l, n) {
    switch (l) {
      case "keyup":
        return Xp.indexOf(n.keyCode) !== -1;
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
  function ii(l) {
    return l = l.detail, typeof l == "object" && "data" in l ? l.data : null;
  }
  var ci = !1;
  function Fh(l, n) {
    switch (l) {
      case "compositionend":
        return ii(n);
      case "keypress":
        return n.which !== 32 ? null : (Os = !0, Wh);
      case "textInput":
        return l = n.data, l === Wh && Os ? null : l;
      default:
        return null;
    }
  }
  function uc(l, n) {
    if (ci)
      return l === "compositionend" || !wf && Bf(l, n) ? (l = Lh(), tc = ro = Au = null, ci = !1, l) : null;
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
        return Hn && n.locale !== "ko" ? null : n.data;
      default:
        return null;
    }
  }
  var Qp = {
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
  function Ds(l) {
    var n = l && l.nodeName && l.nodeName.toLowerCase();
    return n === "input" ? !!Qp[l.type] : n === "textarea";
  }
  function zs(l, n, u, c) {
    co ? oo ? oo.push(c) : oo = [c] : co = c, n = ko(n, "onChange"), 0 < n.length && (u = new Ts(
      "onChange",
      "change",
      null,
      u,
      c
    ), l.push({ event: u, listeners: n }));
  }
  var rn = null, sn = null;
  function Ih(l) {
    wc(l, 0);
  }
  function Nn(l) {
    var n = Af(l);
    if (Wi(n)) return l;
  }
  function Ph(l, n) {
    if (l === "change") return n;
  }
  var ey = !1;
  if (xn) {
    var ic;
    if (xn) {
      var cc = "oninput" in document;
      if (!cc) {
        var ty = document.createElement("div");
        ty.setAttribute("oninput", "return;"), cc = typeof ty.oninput == "function";
      }
      ic = cc;
    } else ic = !1;
    ey = ic && (!document.documentMode || 9 < document.documentMode);
  }
  function so() {
    rn && (rn.detachEvent("onpropertychange", ly), sn = rn = null);
  }
  function ly(l) {
    if (l.propertyName === "value" && Nn(sn)) {
      var n = [];
      zs(
        n,
        sn,
        l,
        vs(l)
      ), fo(Ih, n);
    }
  }
  function Ms(l, n, u) {
    l === "focusin" ? (so(), rn = n, sn = u, rn.attachEvent("onpropertychange", ly)) : l === "focusout" && so();
  }
  function oi(l) {
    if (l === "selectionchange" || l === "keyup" || l === "keydown")
      return Nn(sn);
  }
  function Du(l, n) {
    if (l === "click") return Nn(n);
  }
  function ay(l, n) {
    if (l === "input" || l === "change")
      return Nn(n);
  }
  function ny(l, n) {
    return l === n && (l !== 0 || 1 / l === 1 / n) || l !== l && n !== n;
  }
  var Cl = typeof Object.is == "function" ? Object.is : ny;
  function fi(l, n) {
    if (Cl(l, n)) return !0;
    if (typeof l != "object" || l === null || typeof n != "object" || n === null)
      return !1;
    var u = Object.keys(l), c = Object.keys(n);
    if (u.length !== c.length) return !1;
    for (c = 0; c < u.length; c++) {
      var r = u[c];
      if (!Ki.call(n, r) || !Cl(l[r], n[r]))
        return !1;
    }
    return !0;
  }
  function ri(l) {
    for (; l && l.firstChild; ) l = l.firstChild;
    return l;
  }
  function Nt(l, n) {
    var u = ri(l);
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
      u = ri(u);
    }
  }
  function Yf(l, n) {
    return l && n ? l === n ? !0 : l && l.nodeType === 3 ? !1 : n && n.nodeType === 3 ? Yf(l, n.parentNode) : "contains" in l ? l.contains(n) : l.compareDocumentPosition ? !!(l.compareDocumentPosition(n) & 16) : !1 : !1;
  }
  function uy(l) {
    l = l != null && l.ownerDocument != null && l.ownerDocument.defaultView != null ? l.ownerDocument.defaultView : window;
    for (var n = uo(l.document); n instanceof l.HTMLIFrameElement; ) {
      try {
        var u = typeof n.contentWindow.location.href == "string";
      } catch {
        u = !1;
      }
      if (u) l = n.contentWindow;
      else break;
      n = uo(l.document);
    }
    return n;
  }
  function jf(l) {
    var n = l && l.nodeName && l.nodeName.toLowerCase();
    return n && (n === "input" && (l.type === "text" || l.type === "search" || l.type === "tel" || l.type === "url" || l.type === "password") || n === "textarea" || l.contentEditable === "true");
  }
  var oc = xn && "documentMode" in document && 11 >= document.documentMode, wn = null, dn = null, si = null, fc = !1;
  function _s(l, n, u) {
    var c = u.window === u ? u.document : u.nodeType === 9 ? u : u.ownerDocument;
    fc || wn == null || wn !== uo(c) || (c = wn, "selectionStart" in c && jf(c) ? c = { start: c.selectionStart, end: c.selectionEnd } : (c = (c.ownerDocument && c.ownerDocument.defaultView || window).getSelection(), c = {
      anchorNode: c.anchorNode,
      anchorOffset: c.anchorOffset,
      focusNode: c.focusNode,
      focusOffset: c.focusOffset
    }), si && fi(si, c) || (si = c, c = ko(dn, "onSelect"), 0 < c.length && (n = new Ts(
      "onSelect",
      "select",
      null,
      n,
      u
    ), l.push({ event: n, listeners: c }), n.target = wn)));
  }
  function zu(l, n) {
    var u = {};
    return u[l.toLowerCase()] = n.toLowerCase(), u["Webkit" + l] = "webkit" + n, u["Moz" + l] = "moz" + n, u;
  }
  var rc = {
    animationend: zu("Animation", "AnimationEnd"),
    animationiteration: zu("Animation", "AnimationIteration"),
    animationstart: zu("Animation", "AnimationStart"),
    transitionrun: zu("Transition", "TransitionRun"),
    transitionstart: zu("Transition", "TransitionStart"),
    transitioncancel: zu("Transition", "TransitionCancel"),
    transitionend: zu("Transition", "TransitionEnd")
  }, La = {}, hn = {};
  xn && (hn = document.createElement("div").style, "AnimationEvent" in window || (delete rc.animationend.animation, delete rc.animationiteration.animation, delete rc.animationstart.animation), "TransitionEvent" in window || delete rc.transitionend.transition);
  function qn(l) {
    if (La[l]) return La[l];
    if (!rc[l]) return l;
    var n = rc[l], u;
    for (u in n)
      if (n.hasOwnProperty(u) && u in hn)
        return La[l] = n[u];
    return l;
  }
  var Zp = qn("animationend"), iy = qn("animationiteration"), Kp = qn("animationstart"), cy = qn("transitionrun"), Us = qn("transitionstart"), Jp = qn("transitioncancel"), oy = qn("transitionend"), fy = /* @__PURE__ */ new Map(), ho = "abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(
    " "
  );
  ho.push("scrollEnd");
  function Va(l, n) {
    fy.set(l, n), ti(n, [l]);
  }
  var ry = /* @__PURE__ */ new WeakMap();
  function Da(l, n) {
    if (typeof l == "object" && l !== null) {
      var u = ry.get(l);
      return u !== void 0 ? u : (n = {
        value: l,
        source: n,
        stack: qh(n)
      }, ry.set(l, n), n);
    }
    return {
      value: l,
      source: n,
      stack: qh(n)
    };
  }
  var oa = [], di = 0, Bn = 0;
  function yn() {
    for (var l = di, n = Bn = di = 0; n < l; ) {
      var u = oa[n];
      oa[n++] = null;
      var c = oa[n];
      oa[n++] = null;
      var r = oa[n];
      oa[n++] = null;
      var s = oa[n];
      if (oa[n++] = null, c !== null && r !== null) {
        var y = c.pending;
        y === null ? r.next = r : (r.next = y.next, y.next = r), c.pending = r;
      }
      s !== 0 && mo(u, r, s);
    }
  }
  function hi(l, n, u, c) {
    oa[di++] = l, oa[di++] = n, oa[di++] = u, oa[di++] = c, Bn |= c, l.lanes |= c, l = l.alternate, l !== null && (l.lanes |= c);
  }
  function yo(l, n, u, c) {
    return hi(l, n, u, c), Gf(l);
  }
  function Yn(l, n) {
    return hi(l, null, null, n), Gf(l);
  }
  function mo(l, n, u) {
    l.lanes |= u;
    var c = l.alternate;
    c !== null && (c.lanes |= u);
    for (var r = !1, s = l.return; s !== null; )
      s.childLanes |= u, c = s.alternate, c !== null && (c.childLanes |= u), s.tag === 22 && (l = s.stateNode, l === null || l._visibility & 1 || (r = !0)), l = s, s = s.return;
    return l.tag === 3 ? (s = l.stateNode, r && n !== null && (r = 31 - zl(u), l = s.hiddenUpdates, c = l[r], c === null ? l[r] = [n] : c.push(n), n.lane = u | 536870912), s) : null;
  }
  function Gf(l) {
    if (50 < Vo)
      throw Vo = 0, rm = null, Error(_(185));
    for (var n = l.return; n !== null; )
      l = n, n = l.return;
    return l.tag === 3 ? l.stateNode : null;
  }
  var po = {};
  function kp(l, n, u, c) {
    this.tag = l, this.key = u, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.refCleanup = this.ref = null, this.pendingProps = n, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = c, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function fa(l, n, u, c) {
    return new kp(l, n, u, c);
  }
  function Lf(l) {
    return l = l.prototype, !(!l || !l.isReactComponent);
  }
  function mn(l, n) {
    var u = l.alternate;
    return u === null ? (u = fa(
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
  function ee(l, n, u, c, r, s) {
    var y = 0;
    if (c = l, typeof l == "function") Lf(l) && (y = 1);
    else if (typeof l == "string")
      y = Mv(
        l,
        u,
        ce.current
      ) ? 26 : l === "html" || l === "head" || l === "body" ? 27 : 5;
    else
      e: switch (l) {
        case de:
          return l = fa(31, u, n, r), l.elementType = de, l.lanes = s, l;
        case Oe:
          return Xa(u.children, r, s, n);
        case ot:
          y = 8, r |= 24;
          break;
        case He:
          return l = fa(12, u, n, r | 2), l.elementType = He, l.lanes = s, l;
        case be:
          return l = fa(13, u, n, r), l.elementType = be, l.lanes = s, l;
        case pt:
          return l = fa(19, u, n, r), l.elementType = pt, l.lanes = s, l;
        default:
          if (typeof l == "object" && l !== null)
            switch (l.$$typeof) {
              case Pe:
              case Ke:
                y = 10;
                break e;
              case Ct:
                y = 9;
                break e;
              case At:
                y = 11;
                break e;
              case je:
                y = 14;
                break e;
              case St:
                y = 16, c = null;
                break e;
            }
          y = 29, u = Error(
            _(130, l === null ? "null" : typeof l, "")
          ), c = null;
      }
    return n = fa(y, u, n, r), n.elementType = l, n.type = c, n.lanes = s, n;
  }
  function Xa(l, n, u, c) {
    return l = fa(7, l, c, n), l.lanes = u, l;
  }
  function vo(l, n, u) {
    return l = fa(6, l, null, n), l.lanes = u, l;
  }
  function Vt(l, n, u) {
    return n = fa(
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
  var yi = [], mi = 0, Vf = null, go = 0, Qa = [], ra = 0, Mu = null, pn = 1, Zt = "";
  function it(l, n) {
    yi[mi++] = go, yi[mi++] = Vf, Vf = l, go = n;
  }
  function Cs(l, n, u) {
    Qa[ra++] = pn, Qa[ra++] = Zt, Qa[ra++] = Mu, Mu = l;
    var c = pn;
    l = Zt;
    var r = 32 - zl(c) - 1;
    c &= ~(1 << r), u += 1;
    var s = 32 - zl(n) + r;
    if (30 < s) {
      var y = r - r % 5;
      s = (c & (1 << y) - 1).toString(32), c >>= y, r -= y, pn = 1 << 32 - zl(n) + r | u << r | c, Zt = s + l;
    } else
      pn = 1 << s | u << r | c, Zt = l;
  }
  function sc(l) {
    l.return !== null && (it(l, 1), Cs(l, 1, 0));
  }
  function jn(l) {
    for (; l === Vf; )
      Vf = yi[--mi], yi[mi] = null, go = yi[--mi], yi[mi] = null;
    for (; l === Mu; )
      Mu = Qa[--ra], Qa[ra] = null, Zt = Qa[--ra], Qa[ra] = null, pn = Qa[--ra], Qa[ra] = null;
  }
  var el = null, dt = null, rt = !1, Za = null, Ka = !1, dc = Error(_(519));
  function _u(l) {
    var n = Error(_(418, ""));
    throw To(Da(n, l)), dc;
  }
  function Xf(l) {
    var n = l.stateNode, u = l.type, c = l.memoizedProps;
    switch (n[gl] = l, n[$l] = c, u) {
      case "dialog":
        Ve("cancel", n), Ve("close", n);
        break;
      case "iframe":
      case "object":
      case "embed":
        Ve("load", n);
        break;
      case "video":
      case "audio":
        for (u = 0; u < Mr.length; u++)
          Ve(Mr[u], n);
        break;
      case "source":
        Ve("error", n);
        break;
      case "img":
      case "image":
      case "link":
        Ve("error", n), Ve("load", n);
        break;
      case "details":
        Ve("toggle", n);
        break;
      case "input":
        Ve("invalid", n), ms(
          n,
          c.value,
          c.defaultValue,
          c.checked,
          c.defaultChecked,
          c.type,
          c.name,
          !0
        ), ni(n);
        break;
      case "select":
        Ve("invalid", n);
        break;
      case "textarea":
        Ve("invalid", n), jh(n, c.value, c.defaultValue, c.children), ni(n);
    }
    u = c.children, typeof u != "string" && typeof u != "number" && typeof u != "bigint" || n.textContent === "" + u || c.suppressHydrationWarning === !0 || Am(n.textContent, u) ? (c.popover != null && (Ve("beforetoggle", n), Ve("toggle", n)), c.onScroll != null && Ve("scroll", n), c.onScrollEnd != null && Ve("scrollend", n), c.onClick != null && (n.onclick = Ld), n = !0) : n = !1, n || _u(l);
  }
  function sy(l) {
    for (el = l.return; el; )
      switch (el.tag) {
        case 5:
        case 13:
          Ka = !1;
          return;
        case 27:
        case 3:
          Ka = !0;
          return;
        default:
          el = el.return;
      }
  }
  function bo(l) {
    if (l !== el) return !1;
    if (!rt) return sy(l), rt = !0, !1;
    var n = l.tag, u;
    if ((u = n !== 3 && n !== 27) && ((u = n === 5) && (u = l.type, u = !(u !== "form" && u !== "button") || uu(l.type, l.memoizedProps)), u = !u), u && dt && _u(l), sy(l), n === 13) {
      if (l = l.memoizedState, l = l !== null ? l.dehydrated : null, !l) throw Error(_(317));
      e: {
        for (l = l.nextSibling, n = 0; l; ) {
          if (l.nodeType === 8)
            if (u = l.data, u === "/$") {
              if (n === 0) {
                dt = En(l.nextSibling);
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
      n === 27 ? (n = dt, Hi(l.type) ? (l = Ni, Ni = null, dt = l) : dt = n) : dt = el ? En(l.stateNode.nextSibling) : null;
    return !0;
  }
  function So() {
    dt = el = null, rt = !1;
  }
  function dy() {
    var l = Za;
    return l !== null && (pa === null ? pa = l : pa.push.apply(
      pa,
      l
    ), Za = null), l;
  }
  function To(l) {
    Za === null ? Za = [l] : Za.push(l);
  }
  var Qf = w(null), Uu = null, vn = null;
  function Cu(l, n, u) {
    P(Qf, n._currentValue), n._currentValue = u;
  }
  function Gn(l) {
    l._currentValue = Qf.current, J(Qf);
  }
  function xs(l, n, u) {
    for (; l !== null; ) {
      var c = l.alternate;
      if ((l.childLanes & n) !== n ? (l.childLanes |= n, c !== null && (c.childLanes |= n)) : c !== null && (c.childLanes & n) !== n && (c.childLanes |= n), l === u) break;
      l = l.return;
    }
  }
  function hy(l, n, u, c) {
    var r = l.child;
    for (r !== null && (r.return = l); r !== null; ) {
      var s = r.dependencies;
      if (s !== null) {
        var y = r.child;
        s = s.firstContext;
        e: for (; s !== null; ) {
          var p = s;
          s = r;
          for (var S = 0; S < n.length; S++)
            if (p.context === n[S]) {
              s.lanes |= u, p = s.alternate, p !== null && (p.lanes |= u), xs(
                s.return,
                u,
                l
              ), c || (y = null);
              break e;
            }
          s = p.next;
        }
      } else if (r.tag === 18) {
        if (y = r.return, y === null) throw Error(_(341));
        y.lanes |= u, s = y.alternate, s !== null && (s.lanes |= u), xs(y, u, l), y = null;
      } else y = r.child;
      if (y !== null) y.return = r;
      else
        for (y = r; y !== null; ) {
          if (y === l) {
            y = null;
            break;
          }
          if (r = y.sibling, r !== null) {
            r.return = y.return, y = r;
            break;
          }
          y = y.return;
        }
      r = y;
    }
  }
  function Eo(l, n, u, c) {
    l = null;
    for (var r = n, s = !1; r !== null; ) {
      if (!s) {
        if ((r.flags & 524288) !== 0) s = !0;
        else if ((r.flags & 262144) !== 0) break;
      }
      if (r.tag === 10) {
        var y = r.alternate;
        if (y === null) throw Error(_(387));
        if (y = y.memoizedProps, y !== null) {
          var p = r.type;
          Cl(r.pendingProps.value, y.value) || (l !== null ? l.push(p) : l = [p]);
        }
      } else if (r === cl.current) {
        if (y = r.alternate, y === null) throw Error(_(387));
        y.memoizedState.memoizedState !== r.memoizedState.memoizedState && (l !== null ? l.push(Sa) : l = [Sa]);
      }
      r = r.return;
    }
    l !== null && hy(
      n,
      l,
      u,
      c
    ), n.flags |= 262144;
  }
  function Zf(l) {
    for (l = l.firstContext; l !== null; ) {
      if (!Cl(
        l.context._currentValue,
        l.memoizedValue
      ))
        return !0;
      l = l.next;
    }
    return !1;
  }
  function pi(l) {
    Uu = l, vn = null, l = l.dependencies, l !== null && (l.firstContext = null);
  }
  function bl(l) {
    return yy(Uu, l);
  }
  function Kf(l, n) {
    return Uu === null && pi(l), yy(l, n);
  }
  function yy(l, n) {
    var u = n._currentValue;
    if (n = { context: n, memoizedValue: u, next: null }, vn === null) {
      if (l === null) throw Error(_(308));
      vn = n, l.dependencies = { lanes: 0, firstContext: n }, l.flags |= 524288;
    } else vn = vn.next = n;
    return u;
  }
  var Ro = typeof AbortController < "u" ? AbortController : function() {
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
  }, Hs = H.unstable_scheduleCallback, $p = H.unstable_NormalPriority, rl = {
    $$typeof: Ke,
    Consumer: null,
    Provider: null,
    _currentValue: null,
    _currentValue2: null,
    _threadCount: 0
  };
  function Ao() {
    return {
      controller: new Ro(),
      data: /* @__PURE__ */ new Map(),
      refCount: 0
    };
  }
  function Ln(l) {
    l.refCount--, l.refCount === 0 && Hs($p, function() {
      l.controller.abort();
    });
  }
  var vi = null, Jf = 0, Ja = 0, sl = null;
  function Ns(l, n) {
    if (vi === null) {
      var u = vi = [];
      Jf = 0, Ja = Nc(), sl = {
        status: "pending",
        value: void 0,
        then: function(c) {
          u.push(c);
        }
      };
    }
    return Jf++, n.then(ws, ws), n;
  }
  function ws() {
    if (--Jf === 0 && vi !== null) {
      sl !== null && (sl.status = "fulfilled");
      var l = vi;
      vi = null, Ja = 0, sl = null;
      for (var n = 0; n < l.length; n++) (0, l[n])();
    }
  }
  function Wp(l, n) {
    var u = [], c = {
      status: "pending",
      value: null,
      reason: null,
      then: function(r) {
        u.push(r);
      }
    };
    return l.then(
      function() {
        c.status = "fulfilled", c.value = n;
        for (var r = 0; r < u.length; r++) (0, u[r])(n);
      },
      function(r) {
        for (c.status = "rejected", c.reason = r, r = 0; r < u.length; r++)
          (0, u[r])(void 0);
      }
    ), c;
  }
  var qs = R.S;
  R.S = function(l, n) {
    typeof n == "object" && n !== null && typeof n.then == "function" && Ns(l, n), qs !== null && qs(l, n);
  };
  var Vn = w(null);
  function kf() {
    var l = Vn.current;
    return l !== null ? l : Ut.pooledCache;
  }
  function hc(l, n) {
    n === null ? P(Vn, Vn.current) : P(Vn, n.pool);
  }
  function Bs() {
    var l = kf();
    return l === null ? null : { parent: rl._currentValue, pool: l };
  }
  var gi = Error(_(460)), Ys = Error(_(474)), $f = Error(_(542)), js = { then: function() {
  } };
  function Gs(l) {
    return l = l.status, l === "fulfilled" || l === "rejected";
  }
  function Wf() {
  }
  function my(l, n, u) {
    switch (u = l[u], u === void 0 ? l.push(n) : u !== n && (n.then(Wf, Wf), n = u), n.status) {
      case "fulfilled":
        return n.value;
      case "rejected":
        throw l = n.reason, vy(l), l;
      default:
        if (typeof n.status == "string") n.then(Wf, Wf);
        else {
          if (l = Ut, l !== null && 100 < l.shellSuspendCounter)
            throw Error(_(482));
          l = n, l.status = "pending", l.then(
            function(c) {
              if (n.status === "pending") {
                var r = n;
                r.status = "fulfilled", r.value = c;
              }
            },
            function(c) {
              if (n.status === "pending") {
                var r = n;
                r.status = "rejected", r.reason = c;
              }
            }
          );
        }
        switch (n.status) {
          case "fulfilled":
            return n.value;
          case "rejected":
            throw l = n.reason, vy(l), l;
        }
        throw yc = n, gi;
    }
  }
  var yc = null;
  function py() {
    if (yc === null) throw Error(_(459));
    var l = yc;
    return yc = null, l;
  }
  function vy(l) {
    if (l === gi || l === $f)
      throw Error(_(483));
  }
  var Xn = !1;
  function Ls(l) {
    l.updateQueue = {
      baseState: l.memoizedState,
      firstBaseUpdate: null,
      lastBaseUpdate: null,
      shared: { pending: null, lanes: 0, hiddenCallbacks: null },
      callbacks: null
    };
  }
  function Vs(l, n) {
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
  function Qn(l, n, u) {
    var c = l.updateQueue;
    if (c === null) return null;
    if (c = c.shared, (gt & 2) !== 0) {
      var r = c.pending;
      return r === null ? n.next = n : (n.next = r.next, r.next = n), c.pending = n, n = Gf(l), mo(l, null, u), n;
    }
    return hi(l, c, n, u), Gf(l);
  }
  function mc(l, n, u) {
    if (n = n.updateQueue, n !== null && (n = n.shared, (u & 4194048) !== 0)) {
      var c = n.lanes;
      c &= l.pendingLanes, u |= c, n.lanes = u, Ye(l, u);
    }
  }
  function gy(l, n) {
    var u = l.updateQueue, c = l.alternate;
    if (c !== null && (c = c.updateQueue, u === c)) {
      var r = null, s = null;
      if (u = u.firstBaseUpdate, u !== null) {
        do {
          var y = {
            lane: u.lane,
            tag: u.tag,
            payload: u.payload,
            callback: null,
            next: null
          };
          s === null ? r = s = y : s = s.next = y, u = u.next;
        } while (u !== null);
        s === null ? r = s = n : s = s.next = n;
      } else r = s = n;
      u = {
        baseState: c.baseState,
        firstBaseUpdate: r,
        lastBaseUpdate: s,
        shared: c.shared,
        callbacks: c.callbacks
      }, l.updateQueue = u;
      return;
    }
    l = u.lastBaseUpdate, l === null ? u.firstBaseUpdate = n : l.next = n, u.lastBaseUpdate = n;
  }
  var by = !1;
  function Oo() {
    if (by) {
      var l = sl;
      if (l !== null) throw l;
    }
  }
  function xu(l, n, u, c) {
    by = !1;
    var r = l.updateQueue;
    Xn = !1;
    var s = r.firstBaseUpdate, y = r.lastBaseUpdate, p = r.shared.pending;
    if (p !== null) {
      r.shared.pending = null;
      var S = p, C = S.next;
      S.next = null, y === null ? s = C : y.next = C, y = S;
      var Z = l.alternate;
      Z !== null && (Z = Z.updateQueue, p = Z.lastBaseUpdate, p !== y && (p === null ? Z.firstBaseUpdate = C : p.next = C, Z.lastBaseUpdate = S));
    }
    if (s !== null) {
      var W = r.baseState;
      y = 0, Z = C = S = null, p = s;
      do {
        var N = p.lane & -536870913, B = N !== p.lane;
        if (B ? (lt & N) === N : (c & N) === N) {
          N !== 0 && N === Ja && (by = !0), Z !== null && (Z = Z.next = {
            lane: 0,
            tag: p.tag,
            payload: p.payload,
            callback: null,
            next: null
          });
          e: {
            var Te = l, Ee = p;
            N = n;
            var yt = u;
            switch (Ee.tag) {
              case 1:
                if (Te = Ee.payload, typeof Te == "function") {
                  W = Te.call(yt, W, N);
                  break e;
                }
                W = Te;
                break e;
              case 3:
                Te.flags = Te.flags & -65537 | 128;
              case 0:
                if (Te = Ee.payload, N = typeof Te == "function" ? Te.call(yt, W, N) : Te, N == null) break e;
                W = ie({}, W, N);
                break e;
              case 2:
                Xn = !0;
            }
          }
          N = p.callback, N !== null && (l.flags |= 64, B && (l.flags |= 8192), B = r.callbacks, B === null ? r.callbacks = [N] : B.push(N));
        } else
          B = {
            lane: N,
            tag: p.tag,
            payload: p.payload,
            callback: p.callback,
            next: null
          }, Z === null ? (C = Z = B, S = W) : Z = Z.next = B, y |= N;
        if (p = p.next, p === null) {
          if (p = r.shared.pending, p === null)
            break;
          B = p, p = B.next, B.next = null, r.lastBaseUpdate = B, r.shared.pending = null;
        }
      } while (!0);
      Z === null && (S = W), r.baseState = S, r.firstBaseUpdate = C, r.lastBaseUpdate = Z, s === null && (r.shared.lanes = 0), ju |= y, l.lanes = y, l.memoizedState = W;
    }
  }
  function Xs(l, n) {
    if (typeof l != "function")
      throw Error(_(191, l));
    l.call(n);
  }
  function Ff(l, n) {
    var u = l.callbacks;
    if (u !== null)
      for (l.callbacks = null, l = 0; l < u.length; l++)
        Xs(u[l], n);
  }
  var pc = w(null), If = w(0);
  function Sl(l, n) {
    l = Yu, P(If, l), P(pc, n), Yu = l | n.baseLanes;
  }
  function Do() {
    P(If, Yu), P(pc, pc.current);
  }
  function zo() {
    Yu = If.current, J(pc), J(If);
  }
  var ka = 0, Le = null, vt = null, Xt = null, Pf = !1, za = !1, bi = !1, gn = 0, Ma = 0, Hu = null, Sy = 0;
  function Qt() {
    throw Error(_(321));
  }
  function Qs(l, n) {
    if (n === null) return !1;
    for (var u = 0; u < n.length && u < l.length; u++)
      if (!Cl(l[u], n[u])) return !1;
    return !0;
  }
  function Zs(l, n, u, c, r, s) {
    return ka = s, Le = n, n.memoizedState = null, n.updateQueue = null, n.lanes = 0, R.H = l === null || l.memoizedState === null ? wy : qy, bi = !1, s = u(c, r), bi = !1, za && (s = Ty(
      n,
      u,
      c,
      r
    )), Si(l), s;
  }
  function Si(l) {
    R.H = od;
    var n = vt !== null && vt.next !== null;
    if (ka = 0, Xt = vt = Le = null, Pf = !1, Ma = 0, Hu = null, n) throw Error(_(300));
    l === null || dl || (l = l.dependencies, l !== null && Zf(l) && (dl = !0));
  }
  function Ty(l, n, u, c) {
    Le = l;
    var r = 0;
    do {
      if (za && (Hu = null), Ma = 0, za = !1, 25 <= r) throw Error(_(301));
      if (r += 1, Xt = vt = null, l.updateQueue != null) {
        var s = l.updateQueue;
        s.lastEffect = null, s.events = null, s.stores = null, s.memoCache != null && (s.memoCache.index = 0);
      }
      R.H = Nu, s = n(u, c);
    } while (za);
    return s;
  }
  function Fp() {
    var l = R.H, n = l.useState()[0];
    return n = typeof n.then == "function" ? tr(n) : n, l = l.useState()[0], (vt !== null ? vt.memoizedState : null) !== l && (Le.flags |= 1024), n;
  }
  function Ks() {
    var l = gn !== 0;
    return gn = 0, l;
  }
  function Mo(l, n, u) {
    n.updateQueue = l.updateQueue, n.flags &= -2053, l.lanes &= ~u;
  }
  function Js(l) {
    if (Pf) {
      for (l = l.memoizedState; l !== null; ) {
        var n = l.queue;
        n !== null && (n.pending = null), l = l.next;
      }
      Pf = !1;
    }
    ka = 0, Xt = vt = Le = null, za = !1, Ma = gn = 0, Hu = null;
  }
  function Vl() {
    var l = {
      memoizedState: null,
      baseState: null,
      baseQueue: null,
      queue: null,
      next: null
    };
    return Xt === null ? Le.memoizedState = Xt = l : Xt = Xt.next = l, Xt;
  }
  function Kt() {
    if (vt === null) {
      var l = Le.alternate;
      l = l !== null ? l.memoizedState : null;
    } else l = vt.next;
    var n = Xt === null ? Le.memoizedState : Xt.next;
    if (n !== null)
      Xt = n, vt = l;
    else {
      if (l === null)
        throw Le.alternate === null ? Error(_(467)) : Error(_(310));
      vt = l, l = {
        memoizedState: vt.memoizedState,
        baseState: vt.baseState,
        baseQueue: vt.baseQueue,
        queue: vt.queue,
        next: null
      }, Xt === null ? Le.memoizedState = Xt = l : Xt = Xt.next = l;
    }
    return Xt;
  }
  function er() {
    return { lastEffect: null, events: null, stores: null, memoCache: null };
  }
  function tr(l) {
    var n = Ma;
    return Ma += 1, Hu === null && (Hu = []), l = my(Hu, l, n), n = Le, (Xt === null ? n.memoizedState : Xt.next) === null && (n = n.alternate, R.H = n === null || n.memoizedState === null ? wy : qy), l;
  }
  function nl(l) {
    if (l !== null && typeof l == "object") {
      if (typeof l.then == "function") return tr(l);
      if (l.$$typeof === Ke) return bl(l);
    }
    throw Error(_(438, String(l)));
  }
  function ks(l) {
    var n = null, u = Le.updateQueue;
    if (u !== null && (n = u.memoCache), n == null) {
      var c = Le.alternate;
      c !== null && (c = c.updateQueue, c !== null && (c = c.memoCache, c != null && (n = {
        data: c.data.map(function(r) {
          return r.slice();
        }),
        index: 0
      })));
    }
    if (n == null && (n = { data: [], index: 0 }), u === null && (u = er(), Le.updateQueue = u), u.memoCache = n, u = n.data[n.index], u === void 0)
      for (u = n.data[n.index] = Array(l), c = 0; c < l; c++)
        u[c] = Ot;
    return n.index++, u;
  }
  function Zn(l, n) {
    return typeof n == "function" ? n(l) : n;
  }
  function lr(l) {
    var n = Kt();
    return $s(n, vt, l);
  }
  function $s(l, n, u) {
    var c = l.queue;
    if (c === null) throw Error(_(311));
    c.lastRenderedReducer = u;
    var r = l.baseQueue, s = c.pending;
    if (s !== null) {
      if (r !== null) {
        var y = r.next;
        r.next = s.next, s.next = y;
      }
      n.baseQueue = r = s, c.pending = null;
    }
    if (s = l.baseState, r === null) l.memoizedState = s;
    else {
      n = r.next;
      var p = y = null, S = null, C = n, Z = !1;
      do {
        var W = C.lane & -536870913;
        if (W !== C.lane ? (lt & W) === W : (ka & W) === W) {
          var N = C.revertLane;
          if (N === 0)
            S !== null && (S = S.next = {
              lane: 0,
              revertLane: 0,
              action: C.action,
              hasEagerState: C.hasEagerState,
              eagerState: C.eagerState,
              next: null
            }), W === Ja && (Z = !0);
          else if ((ka & N) === N) {
            C = C.next, N === Ja && (Z = !0);
            continue;
          } else
            W = {
              lane: 0,
              revertLane: C.revertLane,
              action: C.action,
              hasEagerState: C.hasEagerState,
              eagerState: C.eagerState,
              next: null
            }, S === null ? (p = S = W, y = s) : S = S.next = W, Le.lanes |= N, ju |= N;
          W = C.action, bi && u(s, W), s = C.hasEagerState ? C.eagerState : u(s, W);
        } else
          N = {
            lane: W,
            revertLane: C.revertLane,
            action: C.action,
            hasEagerState: C.hasEagerState,
            eagerState: C.eagerState,
            next: null
          }, S === null ? (p = S = N, y = s) : S = S.next = N, Le.lanes |= W, ju |= W;
        C = C.next;
      } while (C !== null && C !== n);
      if (S === null ? y = s : S.next = p, !Cl(s, l.memoizedState) && (dl = !0, Z && (u = sl, u !== null)))
        throw u;
      l.memoizedState = s, l.baseState = y, l.baseQueue = S, c.lastRenderedState = s;
    }
    return r === null && (c.lanes = 0), [l.memoizedState, c.dispatch];
  }
  function Ws(l) {
    var n = Kt(), u = n.queue;
    if (u === null) throw Error(_(311));
    u.lastRenderedReducer = l;
    var c = u.dispatch, r = u.pending, s = n.memoizedState;
    if (r !== null) {
      u.pending = null;
      var y = r = r.next;
      do
        s = l(s, y.action), y = y.next;
      while (y !== r);
      Cl(s, n.memoizedState) || (dl = !0), n.memoizedState = s, n.baseQueue === null && (n.baseState = s), u.lastRenderedState = s;
    }
    return [s, c];
  }
  function ar(l, n, u) {
    var c = Le, r = Kt(), s = rt;
    if (s) {
      if (u === void 0) throw Error(_(407));
      u = u();
    } else u = n();
    var y = !Cl(
      (vt || r).memoizedState,
      u
    );
    y && (r.memoizedState = u, dl = !0), r = r.queue;
    var p = Ry.bind(null, c, r, l);
    if (zt(2048, 8, p, [l]), r.getSnapshot !== n || y || Xt !== null && Xt.memoizedState.tag & 1) {
      if (c.flags |= 2048, da(
        9,
        ir(),
        Ey.bind(
          null,
          c,
          r,
          u,
          n
        ),
        null
      ), Ut === null) throw Error(_(349));
      s || (ka & 124) !== 0 || Fs(c, n, u);
    }
    return u;
  }
  function Fs(l, n, u) {
    l.flags |= 16384, l = { getSnapshot: n, value: u }, n = Le.updateQueue, n === null ? (n = er(), Le.updateQueue = n, n.stores = [l]) : (u = n.stores, u === null ? n.stores = [l] : u.push(l));
  }
  function Ey(l, n, u, c) {
    n.value = u, n.getSnapshot = c, Ay(n) && Is(l);
  }
  function Ry(l, n, u) {
    return u(function() {
      Ay(n) && Is(l);
    });
  }
  function Ay(l) {
    var n = l.getSnapshot;
    l = l.value;
    try {
      var u = n();
      return !Cl(l, u);
    } catch {
      return !0;
    }
  }
  function Is(l) {
    var n = Yn(l, 2);
    n !== null && Ca(n, l, 2);
  }
  function nr(l) {
    var n = Vl();
    if (typeof l == "function") {
      var u = l;
      if (l = u(), bi) {
        ja(!0);
        try {
          u();
        } finally {
          ja(!1);
        }
      }
    }
    return n.memoizedState = n.baseState = l, n.queue = {
      pending: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: Zn,
      lastRenderedState: l
    }, n;
  }
  function Ps(l, n, u, c) {
    return l.baseState = u, $s(
      l,
      vt,
      typeof c == "function" ? c : Zn
    );
  }
  function Ip(l, n, u, c, r) {
    if (Sc(l)) throw Error(_(485));
    if (l = n.action, l !== null) {
      var s = {
        payload: r,
        action: l,
        next: null,
        isTransition: !0,
        status: "pending",
        value: null,
        reason: null,
        listeners: [],
        then: function(y) {
          s.listeners.push(y);
        }
      };
      R.T !== null ? u(!0) : s.isTransition = !1, c(s), u = n.pending, u === null ? (s.next = n.pending = s, ed(n, s)) : (s.next = u.next, n.pending = u.next = s);
    }
  }
  function ed(l, n) {
    var u = n.action, c = n.payload, r = l.state;
    if (n.isTransition) {
      var s = R.T, y = {};
      R.T = y;
      try {
        var p = u(r, c), S = R.S;
        S !== null && S(y, p), ur(l, n, p);
      } catch (C) {
        ld(l, n, C);
      } finally {
        R.T = s;
      }
    } else
      try {
        s = u(r, c), ur(l, n, s);
      } catch (C) {
        ld(l, n, C);
      }
  }
  function ur(l, n, u) {
    u !== null && typeof u == "object" && typeof u.then == "function" ? u.then(
      function(c) {
        td(l, n, c);
      },
      function(c) {
        return ld(l, n, c);
      }
    ) : td(l, n, u);
  }
  function td(l, n, u) {
    n.status = "fulfilled", n.value = u, Oy(n), l.state = u, n = l.pending, n !== null && (u = n.next, u === n ? l.pending = null : (u = u.next, n.next = u, ed(l, u)));
  }
  function ld(l, n, u) {
    var c = l.pending;
    if (l.pending = null, c !== null) {
      c = c.next;
      do
        n.status = "rejected", n.reason = u, Oy(n), n = n.next;
      while (n !== c);
    }
    l.action = null;
  }
  function Oy(l) {
    l = l.listeners;
    for (var n = 0; n < l.length; n++) (0, l[n])();
  }
  function ad(l, n) {
    return n;
  }
  function Dy(l, n) {
    if (rt) {
      var u = Ut.formState;
      if (u !== null) {
        e: {
          var c = Le;
          if (rt) {
            if (dt) {
              t: {
                for (var r = dt, s = Ka; r.nodeType !== 8; ) {
                  if (!s) {
                    r = null;
                    break t;
                  }
                  if (r = En(
                    r.nextSibling
                  ), r === null) {
                    r = null;
                    break t;
                  }
                }
                s = r.data, r = s === "F!" || s === "F" ? r : null;
              }
              if (r) {
                dt = En(
                  r.nextSibling
                ), c = r.data === "F!";
                break e;
              }
            }
            _u(c);
          }
          c = !1;
        }
        c && (n = u[0]);
      }
    }
    return u = Vl(), u.memoizedState = u.baseState = n, c = {
      pending: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: ad,
      lastRenderedState: n
    }, u.queue = c, u = Hy.bind(
      null,
      Le,
      c
    ), c.dispatch = u, c = nr(!1), s = fr.bind(
      null,
      Le,
      !1,
      c.queue
    ), c = Vl(), r = {
      state: n,
      dispatch: null,
      action: l,
      pending: null
    }, c.queue = r, u = Ip.bind(
      null,
      Le,
      r,
      s,
      u
    ), r.dispatch = u, c.memoizedState = l, [n, u, !1];
  }
  function Kn(l) {
    var n = Kt();
    return nd(n, vt, l);
  }
  function nd(l, n, u) {
    if (n = $s(
      l,
      n,
      ad
    )[0], l = lr(Zn)[0], typeof n == "object" && n !== null && typeof n.then == "function")
      try {
        var c = tr(n);
      } catch (y) {
        throw y === gi ? $f : y;
      }
    else c = n;
    n = Kt();
    var r = n.queue, s = r.dispatch;
    return u !== n.memoizedState && (Le.flags |= 2048, da(
      9,
      ir(),
      _g.bind(null, r, u),
      null
    )), [c, s, l];
  }
  function _g(l, n) {
    l.action = n;
  }
  function ud(l) {
    var n = Kt(), u = vt;
    if (u !== null)
      return nd(n, u, l);
    Kt(), n = n.memoizedState, u = Kt();
    var c = u.queue.dispatch;
    return u.memoizedState = l, [n, c, !1];
  }
  function da(l, n, u, c) {
    return l = { tag: l, create: u, deps: c, inst: n, next: null }, n = Le.updateQueue, n === null && (n = er(), Le.updateQueue = n), u = n.lastEffect, u === null ? n.lastEffect = l.next = l : (c = u.next, u.next = l, l.next = c, n.lastEffect = l), l;
  }
  function ir() {
    return { destroy: void 0, resource: void 0 };
  }
  function cr() {
    return Kt().memoizedState;
  }
  function Ti(l, n, u, c) {
    var r = Vl();
    c = c === void 0 ? null : c, Le.flags |= l, r.memoizedState = da(
      1 | n,
      ir(),
      u,
      c
    );
  }
  function zt(l, n, u, c) {
    var r = Kt();
    c = c === void 0 ? null : c;
    var s = r.memoizedState.inst;
    vt !== null && c !== null && Qs(c, vt.memoizedState.deps) ? r.memoizedState = da(n, s, u, c) : (Le.flags |= l, r.memoizedState = da(
      1 | n,
      s,
      u,
      c
    ));
  }
  function Pp(l, n) {
    Ti(8390656, 8, l, n);
  }
  function ev(l, n) {
    zt(2048, 8, l, n);
  }
  function zy(l, n) {
    return zt(4, 2, l, n);
  }
  function bn(l, n) {
    return zt(4, 4, l, n);
  }
  function My(l, n) {
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
  function id(l, n, u) {
    u = u != null ? u.concat([l]) : null, zt(4, 4, My.bind(null, n, l), u);
  }
  function vc() {
  }
  function gc(l, n) {
    var u = Kt();
    n = n === void 0 ? null : n;
    var c = u.memoizedState;
    return n !== null && Qs(n, c[1]) ? c[0] : (u.memoizedState = [l, n], l);
  }
  function _y(l, n) {
    var u = Kt();
    n = n === void 0 ? null : n;
    var c = u.memoizedState;
    if (n !== null && Qs(n, c[1]))
      return c[0];
    if (c = l(), bi) {
      ja(!0);
      try {
        l();
      } finally {
        ja(!1);
      }
    }
    return u.memoizedState = [c, n], c;
  }
  function or(l, n, u) {
    return u === void 0 || (ka & 1073741824) !== 0 ? l.memoizedState = n : (l.memoizedState = u, l = sm(), Le.lanes |= l, ju |= l, u);
  }
  function Uy(l, n, u, c) {
    return Cl(u, n) ? u : pc.current !== null ? (l = or(l, u, c), Cl(l, n) || (dl = !0), l) : (ka & 42) === 0 ? (dl = !0, l.memoizedState = u) : (l = sm(), Le.lanes |= l, ju |= l, n);
  }
  function tv(l, n, u, c, r) {
    var s = X.p;
    X.p = s !== 0 && 8 > s ? s : 8;
    var y = R.T, p = {};
    R.T = p, fr(l, !1, n, u);
    try {
      var S = r(), C = R.S;
      if (C !== null && C(p, S), S !== null && typeof S == "object" && typeof S.then == "function") {
        var Z = Wp(
          S,
          c
        );
        bc(
          l,
          n,
          Z,
          Ua(l)
        );
      } else
        bc(
          l,
          n,
          c,
          Ua(l)
        );
    } catch (W) {
      bc(
        l,
        n,
        { then: function() {
        }, status: "rejected", reason: W },
        Ua()
      );
    } finally {
      X.p = s, R.T = y;
    }
  }
  function Ug() {
  }
  function cd(l, n, u, c) {
    if (l.tag !== 5) throw Error(_(476));
    var r = lv(l).queue;
    tv(
      l,
      r,
      n,
      I,
      u === null ? Ug : function() {
        return _o(l), u(c);
      }
    );
  }
  function lv(l) {
    var n = l.memoizedState;
    if (n !== null) return n;
    n = {
      memoizedState: I,
      baseState: I,
      baseQueue: null,
      queue: {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: Zn,
        lastRenderedState: I
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
        lastRenderedReducer: Zn,
        lastRenderedState: u
      },
      next: null
    }, l.memoizedState = n, l = l.alternate, l !== null && (l.memoizedState = n), n;
  }
  function _o(l) {
    var n = lv(l).next.queue;
    bc(l, n, {}, Ua());
  }
  function $a() {
    return bl(Sa);
  }
  function Cy() {
    return Kt().memoizedState;
  }
  function av() {
    return Kt().memoizedState;
  }
  function nv(l) {
    for (var n = l.return; n !== null; ) {
      switch (n.tag) {
        case 24:
        case 3:
          var u = Ua();
          l = sa(u);
          var c = Qn(n, l, u);
          c !== null && (Ca(c, n, u), mc(c, n, u)), n = { cache: Ao() }, l.payload = n;
          return;
      }
      n = n.return;
    }
  }
  function xy(l, n, u) {
    var c = Ua();
    u = {
      lane: c,
      revertLane: 0,
      action: u,
      hasEagerState: !1,
      eagerState: null,
      next: null
    }, Sc(l) ? uv(n, u) : (u = yo(l, n, u, c), u !== null && (Ca(u, l, c), Ny(u, n, c)));
  }
  function Hy(l, n, u) {
    var c = Ua();
    bc(l, n, u, c);
  }
  function bc(l, n, u, c) {
    var r = {
      lane: c,
      revertLane: 0,
      action: u,
      hasEagerState: !1,
      eagerState: null,
      next: null
    };
    if (Sc(l)) uv(n, r);
    else {
      var s = l.alternate;
      if (l.lanes === 0 && (s === null || s.lanes === 0) && (s = n.lastRenderedReducer, s !== null))
        try {
          var y = n.lastRenderedState, p = s(y, u);
          if (r.hasEagerState = !0, r.eagerState = p, Cl(p, y))
            return hi(l, n, r, 0), Ut === null && yn(), !1;
        } catch {
        } finally {
        }
      if (u = yo(l, n, r, c), u !== null)
        return Ca(u, l, c), Ny(u, n, c), !0;
    }
    return !1;
  }
  function fr(l, n, u, c) {
    if (c = {
      lane: 2,
      revertLane: Nc(),
      action: c,
      hasEagerState: !1,
      eagerState: null,
      next: null
    }, Sc(l)) {
      if (n) throw Error(_(479));
    } else
      n = yo(
        l,
        u,
        c,
        2
      ), n !== null && Ca(n, l, 2);
  }
  function Sc(l) {
    var n = l.alternate;
    return l === Le || n !== null && n === Le;
  }
  function uv(l, n) {
    za = Pf = !0;
    var u = l.pending;
    u === null ? n.next = n : (n.next = u.next, u.next = n), l.pending = n;
  }
  function Ny(l, n, u) {
    if ((u & 4194048) !== 0) {
      var c = n.lanes;
      c &= l.pendingLanes, u |= c, n.lanes = u, Ye(l, u);
    }
  }
  var od = {
    readContext: bl,
    use: nl,
    useCallback: Qt,
    useContext: Qt,
    useEffect: Qt,
    useImperativeHandle: Qt,
    useLayoutEffect: Qt,
    useInsertionEffect: Qt,
    useMemo: Qt,
    useReducer: Qt,
    useRef: Qt,
    useState: Qt,
    useDebugValue: Qt,
    useDeferredValue: Qt,
    useTransition: Qt,
    useSyncExternalStore: Qt,
    useId: Qt,
    useHostTransitionStatus: Qt,
    useFormState: Qt,
    useActionState: Qt,
    useOptimistic: Qt,
    useMemoCache: Qt,
    useCacheRefresh: Qt
  }, wy = {
    readContext: bl,
    use: nl,
    useCallback: function(l, n) {
      return Vl().memoizedState = [
        l,
        n === void 0 ? null : n
      ], l;
    },
    useContext: bl,
    useEffect: Pp,
    useImperativeHandle: function(l, n, u) {
      u = u != null ? u.concat([l]) : null, Ti(
        4194308,
        4,
        My.bind(null, n, l),
        u
      );
    },
    useLayoutEffect: function(l, n) {
      return Ti(4194308, 4, l, n);
    },
    useInsertionEffect: function(l, n) {
      Ti(4, 2, l, n);
    },
    useMemo: function(l, n) {
      var u = Vl();
      n = n === void 0 ? null : n;
      var c = l();
      if (bi) {
        ja(!0);
        try {
          l();
        } finally {
          ja(!1);
        }
      }
      return u.memoizedState = [c, n], c;
    },
    useReducer: function(l, n, u) {
      var c = Vl();
      if (u !== void 0) {
        var r = u(n);
        if (bi) {
          ja(!0);
          try {
            u(n);
          } finally {
            ja(!1);
          }
        }
      } else r = n;
      return c.memoizedState = c.baseState = r, l = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: l,
        lastRenderedState: r
      }, c.queue = l, l = l.dispatch = xy.bind(
        null,
        Le,
        l
      ), [c.memoizedState, l];
    },
    useRef: function(l) {
      var n = Vl();
      return l = { current: l }, n.memoizedState = l;
    },
    useState: function(l) {
      l = nr(l);
      var n = l.queue, u = Hy.bind(null, Le, n);
      return n.dispatch = u, [l.memoizedState, u];
    },
    useDebugValue: vc,
    useDeferredValue: function(l, n) {
      var u = Vl();
      return or(u, l, n);
    },
    useTransition: function() {
      var l = nr(!1);
      return l = tv.bind(
        null,
        Le,
        l.queue,
        !0,
        !1
      ), Vl().memoizedState = l, [!1, l];
    },
    useSyncExternalStore: function(l, n, u) {
      var c = Le, r = Vl();
      if (rt) {
        if (u === void 0)
          throw Error(_(407));
        u = u();
      } else {
        if (u = n(), Ut === null)
          throw Error(_(349));
        (lt & 124) !== 0 || Fs(c, n, u);
      }
      r.memoizedState = u;
      var s = { value: u, getSnapshot: n };
      return r.queue = s, Pp(Ry.bind(null, c, s, l), [
        l
      ]), c.flags |= 2048, da(
        9,
        ir(),
        Ey.bind(
          null,
          c,
          s,
          u,
          n
        ),
        null
      ), u;
    },
    useId: function() {
      var l = Vl(), n = Ut.identifierPrefix;
      if (rt) {
        var u = Zt, c = pn;
        u = (c & ~(1 << 32 - zl(c) - 1)).toString(32) + u, n = "«" + n + "R" + u, u = gn++, 0 < u && (n += "H" + u.toString(32)), n += "»";
      } else
        u = Sy++, n = "«" + n + "r" + u.toString(32) + "»";
      return l.memoizedState = n;
    },
    useHostTransitionStatus: $a,
    useFormState: Dy,
    useActionState: Dy,
    useOptimistic: function(l) {
      var n = Vl();
      n.memoizedState = n.baseState = l;
      var u = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: null,
        lastRenderedState: null
      };
      return n.queue = u, n = fr.bind(
        null,
        Le,
        !0,
        u
      ), u.dispatch = n, [l, n];
    },
    useMemoCache: ks,
    useCacheRefresh: function() {
      return Vl().memoizedState = nv.bind(
        null,
        Le
      );
    }
  }, qy = {
    readContext: bl,
    use: nl,
    useCallback: gc,
    useContext: bl,
    useEffect: ev,
    useImperativeHandle: id,
    useInsertionEffect: zy,
    useLayoutEffect: bn,
    useMemo: _y,
    useReducer: lr,
    useRef: cr,
    useState: function() {
      return lr(Zn);
    },
    useDebugValue: vc,
    useDeferredValue: function(l, n) {
      var u = Kt();
      return Uy(
        u,
        vt.memoizedState,
        l,
        n
      );
    },
    useTransition: function() {
      var l = lr(Zn)[0], n = Kt().memoizedState;
      return [
        typeof l == "boolean" ? l : tr(l),
        n
      ];
    },
    useSyncExternalStore: ar,
    useId: Cy,
    useHostTransitionStatus: $a,
    useFormState: Kn,
    useActionState: Kn,
    useOptimistic: function(l, n) {
      var u = Kt();
      return Ps(u, vt, l, n);
    },
    useMemoCache: ks,
    useCacheRefresh: av
  }, Nu = {
    readContext: bl,
    use: nl,
    useCallback: gc,
    useContext: bl,
    useEffect: ev,
    useImperativeHandle: id,
    useInsertionEffect: zy,
    useLayoutEffect: bn,
    useMemo: _y,
    useReducer: Ws,
    useRef: cr,
    useState: function() {
      return Ws(Zn);
    },
    useDebugValue: vc,
    useDeferredValue: function(l, n) {
      var u = Kt();
      return vt === null ? or(u, l, n) : Uy(
        u,
        vt.memoizedState,
        l,
        n
      );
    },
    useTransition: function() {
      var l = Ws(Zn)[0], n = Kt().memoizedState;
      return [
        typeof l == "boolean" ? l : tr(l),
        n
      ];
    },
    useSyncExternalStore: ar,
    useId: Cy,
    useHostTransitionStatus: $a,
    useFormState: ud,
    useActionState: ud,
    useOptimistic: function(l, n) {
      var u = Kt();
      return vt !== null ? Ps(u, vt, l, n) : (u.baseState = l, [l, u.queue.dispatch]);
    },
    useMemoCache: ks,
    useCacheRefresh: av
  }, Tc = null, Uo = 0;
  function fd(l) {
    var n = Uo;
    return Uo += 1, Tc === null && (Tc = []), my(Tc, l, n);
  }
  function Ec(l, n) {
    n = n.props.ref, l.ref = n !== void 0 ? n : null;
  }
  function Xl(l, n) {
    throw n.$$typeof === K ? Error(_(525)) : (l = Object.prototype.toString.call(n), Error(
      _(
        31,
        l === "[object Object]" ? "object with keys {" + Object.keys(n).join(", ") + "}" : l
      )
    ));
  }
  function By(l) {
    var n = l._init;
    return n(l._payload);
  }
  function ha(l) {
    function n(M, O) {
      if (l) {
        var U = M.deletions;
        U === null ? (M.deletions = [O], M.flags |= 16) : U.push(O);
      }
    }
    function u(M, O) {
      if (!l) return null;
      for (; O !== null; )
        n(M, O), O = O.sibling;
      return null;
    }
    function c(M) {
      for (var O = /* @__PURE__ */ new Map(); M !== null; )
        M.key !== null ? O.set(M.key, M) : O.set(M.index, M), M = M.sibling;
      return O;
    }
    function r(M, O) {
      return M = mn(M, O), M.index = 0, M.sibling = null, M;
    }
    function s(M, O, U) {
      return M.index = U, l ? (U = M.alternate, U !== null ? (U = U.index, U < O ? (M.flags |= 67108866, O) : U) : (M.flags |= 67108866, O)) : (M.flags |= 1048576, O);
    }
    function y(M) {
      return l && M.alternate === null && (M.flags |= 67108866), M;
    }
    function p(M, O, U, $) {
      return O === null || O.tag !== 6 ? (O = vo(U, M.mode, $), O.return = M, O) : (O = r(O, U), O.return = M, O);
    }
    function S(M, O, U, $) {
      var se = U.type;
      return se === Oe ? Z(
        M,
        O,
        U.props.children,
        $,
        U.key
      ) : O !== null && (O.elementType === se || typeof se == "object" && se !== null && se.$$typeof === St && By(se) === O.type) ? (O = r(O, U.props), Ec(O, U), O.return = M, O) : (O = ee(
        U.type,
        U.key,
        U.props,
        null,
        M.mode,
        $
      ), Ec(O, U), O.return = M, O);
    }
    function C(M, O, U, $) {
      return O === null || O.tag !== 4 || O.stateNode.containerInfo !== U.containerInfo || O.stateNode.implementation !== U.implementation ? (O = Vt(U, M.mode, $), O.return = M, O) : (O = r(O, U.children || []), O.return = M, O);
    }
    function Z(M, O, U, $, se) {
      return O === null || O.tag !== 7 ? (O = Xa(
        U,
        M.mode,
        $,
        se
      ), O.return = M, O) : (O = r(O, U), O.return = M, O);
    }
    function W(M, O, U) {
      if (typeof O == "string" && O !== "" || typeof O == "number" || typeof O == "bigint")
        return O = vo(
          "" + O,
          M.mode,
          U
        ), O.return = M, O;
      if (typeof O == "object" && O !== null) {
        switch (O.$$typeof) {
          case D:
            return U = ee(
              O.type,
              O.key,
              O.props,
              null,
              M.mode,
              U
            ), Ec(U, O), U.return = M, U;
          case ue:
            return O = Vt(
              O,
              M.mode,
              U
            ), O.return = M, O;
          case St:
            var $ = O._init;
            return O = $(O._payload), W(M, O, U);
        }
        if (le(O) || ze(O))
          return O = Xa(
            O,
            M.mode,
            U,
            null
          ), O.return = M, O;
        if (typeof O.then == "function")
          return W(M, fd(O), U);
        if (O.$$typeof === Ke)
          return W(
            M,
            Kf(M, O),
            U
          );
        Xl(M, O);
      }
      return null;
    }
    function N(M, O, U, $) {
      var se = O !== null ? O.key : null;
      if (typeof U == "string" && U !== "" || typeof U == "number" || typeof U == "bigint")
        return se !== null ? null : p(M, O, "" + U, $);
      if (typeof U == "object" && U !== null) {
        switch (U.$$typeof) {
          case D:
            return U.key === se ? S(M, O, U, $) : null;
          case ue:
            return U.key === se ? C(M, O, U, $) : null;
          case St:
            return se = U._init, U = se(U._payload), N(M, O, U, $);
        }
        if (le(U) || ze(U))
          return se !== null ? null : Z(M, O, U, $, null);
        if (typeof U.then == "function")
          return N(
            M,
            O,
            fd(U),
            $
          );
        if (U.$$typeof === Ke)
          return N(
            M,
            O,
            Kf(M, U),
            $
          );
        Xl(M, U);
      }
      return null;
    }
    function B(M, O, U, $, se) {
      if (typeof $ == "string" && $ !== "" || typeof $ == "number" || typeof $ == "bigint")
        return M = M.get(U) || null, p(O, M, "" + $, se);
      if (typeof $ == "object" && $ !== null) {
        switch ($.$$typeof) {
          case D:
            return M = M.get(
              $.key === null ? U : $.key
            ) || null, S(O, M, $, se);
          case ue:
            return M = M.get(
              $.key === null ? U : $.key
            ) || null, C(O, M, $, se);
          case St:
            var We = $._init;
            return $ = We($._payload), B(
              M,
              O,
              U,
              $,
              se
            );
        }
        if (le($) || ze($))
          return M = M.get(U) || null, Z(O, M, $, se, null);
        if (typeof $.then == "function")
          return B(
            M,
            O,
            U,
            fd($),
            se
          );
        if ($.$$typeof === Ke)
          return B(
            M,
            O,
            U,
            Kf(O, $),
            se
          );
        Xl(O, $);
      }
      return null;
    }
    function Te(M, O, U, $) {
      for (var se = null, We = null, Se = O, _e = O = 0, Rl = null; Se !== null && _e < U.length; _e++) {
        Se.index > _e ? (Rl = Se, Se = null) : Rl = Se.sibling;
        var ft = N(
          M,
          Se,
          U[_e],
          $
        );
        if (ft === null) {
          Se === null && (Se = Rl);
          break;
        }
        l && Se && ft.alternate === null && n(M, Se), O = s(ft, O, _e), We === null ? se = ft : We.sibling = ft, We = ft, Se = Rl;
      }
      if (_e === U.length)
        return u(M, Se), rt && it(M, _e), se;
      if (Se === null) {
        for (; _e < U.length; _e++)
          Se = W(M, U[_e], $), Se !== null && (O = s(
            Se,
            O,
            _e
          ), We === null ? se = Se : We.sibling = Se, We = Se);
        return rt && it(M, _e), se;
      }
      for (Se = c(Se); _e < U.length; _e++)
        Rl = B(
          Se,
          M,
          _e,
          U[_e],
          $
        ), Rl !== null && (l && Rl.alternate !== null && Se.delete(
          Rl.key === null ? _e : Rl.key
        ), O = s(
          Rl,
          O,
          _e
        ), We === null ? se = Rl : We.sibling = Rl, We = Rl);
      return l && Se.forEach(function(ji) {
        return n(M, ji);
      }), rt && it(M, _e), se;
    }
    function Ee(M, O, U, $) {
      if (U == null) throw Error(_(151));
      for (var se = null, We = null, Se = O, _e = O = 0, Rl = null, ft = U.next(); Se !== null && !ft.done; _e++, ft = U.next()) {
        Se.index > _e ? (Rl = Se, Se = null) : Rl = Se.sibling;
        var ji = N(M, Se, ft.value, $);
        if (ji === null) {
          Se === null && (Se = Rl);
          break;
        }
        l && Se && ji.alternate === null && n(M, Se), O = s(ji, O, _e), We === null ? se = ji : We.sibling = ji, We = ji, Se = Rl;
      }
      if (ft.done)
        return u(M, Se), rt && it(M, _e), se;
      if (Se === null) {
        for (; !ft.done; _e++, ft = U.next())
          ft = W(M, ft.value, $), ft !== null && (O = s(ft, O, _e), We === null ? se = ft : We.sibling = ft, We = ft);
        return rt && it(M, _e), se;
      }
      for (Se = c(Se); !ft.done; _e++, ft = U.next())
        ft = B(Se, M, _e, ft.value, $), ft !== null && (l && ft.alternate !== null && Se.delete(ft.key === null ? _e : ft.key), O = s(ft, O, _e), We === null ? se = ft : We.sibling = ft, We = ft);
      return l && Se.forEach(function(Vg) {
        return n(M, Vg);
      }), rt && it(M, _e), se;
    }
    function yt(M, O, U, $) {
      if (typeof U == "object" && U !== null && U.type === Oe && U.key === null && (U = U.props.children), typeof U == "object" && U !== null) {
        switch (U.$$typeof) {
          case D:
            e: {
              for (var se = U.key; O !== null; ) {
                if (O.key === se) {
                  if (se = U.type, se === Oe) {
                    if (O.tag === 7) {
                      u(
                        M,
                        O.sibling
                      ), $ = r(
                        O,
                        U.props.children
                      ), $.return = M, M = $;
                      break e;
                    }
                  } else if (O.elementType === se || typeof se == "object" && se !== null && se.$$typeof === St && By(se) === O.type) {
                    u(
                      M,
                      O.sibling
                    ), $ = r(O, U.props), Ec($, U), $.return = M, M = $;
                    break e;
                  }
                  u(M, O);
                  break;
                } else n(M, O);
                O = O.sibling;
              }
              U.type === Oe ? ($ = Xa(
                U.props.children,
                M.mode,
                $,
                U.key
              ), $.return = M, M = $) : ($ = ee(
                U.type,
                U.key,
                U.props,
                null,
                M.mode,
                $
              ), Ec($, U), $.return = M, M = $);
            }
            return y(M);
          case ue:
            e: {
              for (se = U.key; O !== null; ) {
                if (O.key === se)
                  if (O.tag === 4 && O.stateNode.containerInfo === U.containerInfo && O.stateNode.implementation === U.implementation) {
                    u(
                      M,
                      O.sibling
                    ), $ = r(O, U.children || []), $.return = M, M = $;
                    break e;
                  } else {
                    u(M, O);
                    break;
                  }
                else n(M, O);
                O = O.sibling;
              }
              $ = Vt(U, M.mode, $), $.return = M, M = $;
            }
            return y(M);
          case St:
            return se = U._init, U = se(U._payload), yt(
              M,
              O,
              U,
              $
            );
        }
        if (le(U))
          return Te(
            M,
            O,
            U,
            $
          );
        if (ze(U)) {
          if (se = ze(U), typeof se != "function") throw Error(_(150));
          return U = se.call(U), Ee(
            M,
            O,
            U,
            $
          );
        }
        if (typeof U.then == "function")
          return yt(
            M,
            O,
            fd(U),
            $
          );
        if (U.$$typeof === Ke)
          return yt(
            M,
            O,
            Kf(M, U),
            $
          );
        Xl(M, U);
      }
      return typeof U == "string" && U !== "" || typeof U == "number" || typeof U == "bigint" ? (U = "" + U, O !== null && O.tag === 6 ? (u(M, O.sibling), $ = r(O, U), $.return = M, M = $) : (u(M, O), $ = vo(U, M.mode, $), $.return = M, M = $), y(M)) : u(M, O);
    }
    return function(M, O, U, $) {
      try {
        Uo = 0;
        var se = yt(
          M,
          O,
          U,
          $
        );
        return Tc = null, se;
      } catch (Se) {
        if (Se === gi || Se === $f) throw Se;
        var We = fa(29, Se, null, M.mode);
        return We.lanes = $, We.return = M, We;
      } finally {
      }
    };
  }
  var Rc = ha(!0), Jn = ha(!1), _a = w(null), Ql = null;
  function wu(l) {
    var n = l.alternate;
    P(Mt, Mt.current & 1), P(_a, l), Ql === null && (n === null || pc.current !== null || n.memoizedState !== null) && (Ql = l);
  }
  function kn(l) {
    if (l.tag === 22) {
      if (P(Mt, Mt.current), P(_a, l), Ql === null) {
        var n = l.alternate;
        n !== null && n.memoizedState !== null && (Ql = l);
      }
    } else $n();
  }
  function $n() {
    P(Mt, Mt.current), P(_a, _a.current);
  }
  function Sn(l) {
    J(_a), Ql === l && (Ql = null), J(Mt);
  }
  var Mt = w(0);
  function rr(l) {
    for (var n = l; n !== null; ) {
      if (n.tag === 13) {
        var u = n.memoizedState;
        if (u !== null && (u = u.dehydrated, u === null || u.data === "$?" || Hr(u)))
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
  function Ei(l, n, u, c) {
    n = l.memoizedState, u = u(c, n), u = u == null ? n : ie({}, n, u), l.memoizedState = u, l.lanes === 0 && (l.updateQueue.baseState = u);
  }
  var rd = {
    enqueueSetState: function(l, n, u) {
      l = l._reactInternals;
      var c = Ua(), r = sa(c);
      r.payload = n, u != null && (r.callback = u), n = Qn(l, r, c), n !== null && (Ca(n, l, c), mc(n, l, c));
    },
    enqueueReplaceState: function(l, n, u) {
      l = l._reactInternals;
      var c = Ua(), r = sa(c);
      r.tag = 1, r.payload = n, u != null && (r.callback = u), n = Qn(l, r, c), n !== null && (Ca(n, l, c), mc(n, l, c));
    },
    enqueueForceUpdate: function(l, n) {
      l = l._reactInternals;
      var u = Ua(), c = sa(u);
      c.tag = 2, n != null && (c.callback = n), n = Qn(l, c, u), n !== null && (Ca(n, l, u), mc(n, l, u));
    }
  };
  function Co(l, n, u, c, r, s, y) {
    return l = l.stateNode, typeof l.shouldComponentUpdate == "function" ? l.shouldComponentUpdate(c, s, y) : n.prototype && n.prototype.isPureReactComponent ? !fi(u, c) || !fi(r, s) : !0;
  }
  function Ac(l, n, u, c) {
    l = n.state, typeof n.componentWillReceiveProps == "function" && n.componentWillReceiveProps(u, c), typeof n.UNSAFE_componentWillReceiveProps == "function" && n.UNSAFE_componentWillReceiveProps(u, c), n.state !== l && rd.enqueueReplaceState(n, n.state, null);
  }
  function Ri(l, n) {
    var u = n;
    if ("ref" in n) {
      u = {};
      for (var c in n)
        c !== "ref" && (u[c] = n[c]);
    }
    if (l = l.defaultProps) {
      u === n && (u = ie({}, u));
      for (var r in l)
        u[r] === void 0 && (u[r] = l[r]);
    }
    return u;
  }
  var sr = typeof reportError == "function" ? reportError : function(l) {
    if (typeof window == "object" && typeof window.ErrorEvent == "function") {
      var n = new window.ErrorEvent("error", {
        bubbles: !0,
        cancelable: !0,
        message: typeof l == "object" && l !== null && typeof l.message == "string" ? String(l.message) : String(l),
        error: l
      });
      if (!window.dispatchEvent(n)) return;
    } else if (typeof Pt == "object" && typeof Pt.emit == "function") {
      Pt.emit("uncaughtException", l);
      return;
    }
    console.error(l);
  };
  function xo(l) {
    sr(l);
  }
  function Yy(l) {
    console.error(l);
  }
  function dr(l) {
    sr(l);
  }
  function hr(l, n) {
    try {
      var u = l.onUncaughtError;
      u(n.value, { componentStack: n.stack });
    } catch (c) {
      setTimeout(function() {
        throw c;
      });
    }
  }
  function jy(l, n, u) {
    try {
      var c = l.onCaughtError;
      c(u.value, {
        componentStack: u.stack,
        errorBoundary: n.tag === 1 ? n.stateNode : null
      });
    } catch (r) {
      setTimeout(function() {
        throw r;
      });
    }
  }
  function Gy(l, n, u) {
    return u = sa(u), u.tag = 3, u.payload = { element: null }, u.callback = function() {
      hr(l, n);
    }, u;
  }
  function Ly(l) {
    return l = sa(l), l.tag = 3, l;
  }
  function ya(l, n, u, c) {
    var r = u.type.getDerivedStateFromError;
    if (typeof r == "function") {
      var s = c.value;
      l.payload = function() {
        return r(s);
      }, l.callback = function() {
        jy(n, u, c);
      };
    }
    var y = u.stateNode;
    y !== null && typeof y.componentDidCatch == "function" && (l.callback = function() {
      jy(n, u, c), typeof r != "function" && (zi === null ? zi = /* @__PURE__ */ new Set([this]) : zi.add(this));
      var p = c.stack;
      this.componentDidCatch(c.value, {
        componentStack: p !== null ? p : ""
      });
    });
  }
  function iv(l, n, u, c, r) {
    if (u.flags |= 32768, c !== null && typeof c == "object" && typeof c.then == "function") {
      if (n = u.alternate, n !== null && Eo(
        n,
        u,
        r,
        !0
      ), u = _a.current, u !== null) {
        switch (u.tag) {
          case 13:
            return Ql === null ? Hc() : u.alternate === null && Wt === 0 && (Wt = 3), u.flags &= -257, u.flags |= 65536, u.lanes = r, c === js ? u.flags |= 16384 : (n = u.updateQueue, n === null ? u.updateQueue = /* @__PURE__ */ new Set([c]) : n.add(c), qd(l, c, r)), !1;
          case 22:
            return u.flags |= 65536, c === js ? u.flags |= 16384 : (n = u.updateQueue, n === null ? (n = {
              transitions: null,
              markerInstances: null,
              retryQueue: /* @__PURE__ */ new Set([c])
            }, u.updateQueue = n) : (u = n.retryQueue, u === null ? n.retryQueue = /* @__PURE__ */ new Set([c]) : u.add(c)), qd(l, c, r)), !1;
        }
        throw Error(_(435, u.tag));
      }
      return qd(l, c, r), Hc(), !1;
    }
    if (rt)
      return n = _a.current, n !== null ? ((n.flags & 65536) === 0 && (n.flags |= 256), n.flags |= 65536, n.lanes = r, c !== dc && (l = Error(_(422), { cause: c }), To(Da(l, u)))) : (c !== dc && (n = Error(_(423), {
        cause: c
      }), To(
        Da(n, u)
      )), l = l.current.alternate, l.flags |= 65536, r &= -r, l.lanes |= r, c = Da(c, u), r = Gy(
        l.stateNode,
        c,
        r
      ), gy(l, r), Wt !== 4 && (Wt = 2)), !1;
    var s = Error(_(520), { cause: c });
    if (s = Da(s, u), jo === null ? jo = [s] : jo.push(s), Wt !== 4 && (Wt = 2), n === null) return !0;
    c = Da(c, u), u = n;
    do {
      switch (u.tag) {
        case 3:
          return u.flags |= 65536, l = r & -r, u.lanes |= l, l = Gy(u.stateNode, c, l), gy(u, l), !1;
        case 1:
          if (n = u.type, s = u.stateNode, (u.flags & 128) === 0 && (typeof n.getDerivedStateFromError == "function" || s !== null && typeof s.componentDidCatch == "function" && (zi === null || !zi.has(s))))
            return u.flags |= 65536, r &= -r, u.lanes |= r, r = Ly(r), ya(
              r,
              l,
              u,
              c
            ), gy(u, r), !1;
      }
      u = u.return;
    } while (u !== null);
    return !1;
  }
  var Jt = Error(_(461)), dl = !1;
  function Tl(l, n, u, c) {
    n.child = l === null ? Jn(n, null, u, c) : Rc(
      n,
      l.child,
      u,
      c
    );
  }
  function cv(l, n, u, c, r) {
    u = u.render;
    var s = n.ref;
    if ("ref" in c) {
      var y = {};
      for (var p in c)
        p !== "ref" && (y[p] = c[p]);
    } else y = c;
    return pi(n), c = Zs(
      l,
      n,
      u,
      y,
      s,
      r
    ), p = Ks(), l !== null && !dl ? (Mo(l, n, r), Wn(l, n, r)) : (rt && p && sc(n), n.flags |= 1, Tl(l, n, c, r), n.child);
  }
  function qu(l, n, u, c, r) {
    if (l === null) {
      var s = u.type;
      return typeof s == "function" && !Lf(s) && s.defaultProps === void 0 && u.compare === null ? (n.tag = 15, n.type = s, Oc(
        l,
        n,
        s,
        c,
        r
      )) : (l = ee(
        u.type,
        null,
        c,
        n,
        n.mode,
        r
      ), l.ref = n.ref, l.return = n, n.child = l);
    }
    if (s = l.child, !Sd(l, r)) {
      var y = s.memoizedProps;
      if (u = u.compare, u = u !== null ? u : fi, u(y, c) && l.ref === n.ref)
        return Wn(l, n, r);
    }
    return n.flags |= 1, l = mn(s, c), l.ref = n.ref, l.return = n, n.child = l;
  }
  function Oc(l, n, u, c, r) {
    if (l !== null) {
      var s = l.memoizedProps;
      if (fi(s, c) && l.ref === n.ref)
        if (dl = !1, n.pendingProps = c = s, Sd(l, r))
          (l.flags & 131072) !== 0 && (dl = !0);
        else
          return n.lanes = l.lanes, Wn(l, n, r);
    }
    return dd(
      l,
      n,
      u,
      c,
      r
    );
  }
  function sd(l, n, u) {
    var c = n.pendingProps, r = c.children, s = l !== null ? l.memoizedState : null;
    if (c.mode === "hidden") {
      if ((n.flags & 128) !== 0) {
        if (c = s !== null ? s.baseLanes | u : u, l !== null) {
          for (r = n.child = l.child, s = 0; r !== null; )
            s = s | r.lanes | r.childLanes, r = r.sibling;
          n.childLanes = s & ~c;
        } else n.childLanes = 0, n.child = null;
        return Dc(
          l,
          n,
          c,
          u
        );
      }
      if ((u & 536870912) !== 0)
        n.memoizedState = { baseLanes: 0, cachePool: null }, l !== null && hc(
          n,
          s !== null ? s.cachePool : null
        ), s !== null ? Sl(n, s) : Do(), kn(n);
      else
        return n.lanes = n.childLanes = 536870912, Dc(
          l,
          n,
          s !== null ? s.baseLanes | u : u,
          u
        );
    } else
      s !== null ? (hc(n, s.cachePool), Sl(n, s), $n(), n.memoizedState = null) : (l !== null && hc(n, null), Do(), $n());
    return Tl(l, n, r, u), n.child;
  }
  function Dc(l, n, u, c) {
    var r = kf();
    return r = r === null ? null : { parent: rl._currentValue, pool: r }, n.memoizedState = {
      baseLanes: u,
      cachePool: r
    }, l !== null && hc(n, null), Do(), kn(n), l !== null && Eo(l, n, c, !0), null;
  }
  function yr(l, n) {
    var u = n.ref;
    if (u === null)
      l !== null && l.ref !== null && (n.flags |= 4194816);
    else {
      if (typeof u != "function" && typeof u != "object")
        throw Error(_(284));
      (l === null || l.ref !== u) && (n.flags |= 4194816);
    }
  }
  function dd(l, n, u, c, r) {
    return pi(n), u = Zs(
      l,
      n,
      u,
      c,
      void 0,
      r
    ), c = Ks(), l !== null && !dl ? (Mo(l, n, r), Wn(l, n, r)) : (rt && c && sc(n), n.flags |= 1, Tl(l, n, u, r), n.child);
  }
  function Vy(l, n, u, c, r, s) {
    return pi(n), n.updateQueue = null, u = Ty(
      n,
      c,
      u,
      r
    ), Si(l), c = Ks(), l !== null && !dl ? (Mo(l, n, s), Wn(l, n, s)) : (rt && c && sc(n), n.flags |= 1, Tl(l, n, u, s), n.child);
  }
  function hd(l, n, u, c, r) {
    if (pi(n), n.stateNode === null) {
      var s = po, y = u.contextType;
      typeof y == "object" && y !== null && (s = bl(y)), s = new u(c, s), n.memoizedState = s.state !== null && s.state !== void 0 ? s.state : null, s.updater = rd, n.stateNode = s, s._reactInternals = n, s = n.stateNode, s.props = c, s.state = n.memoizedState, s.refs = {}, Ls(n), y = u.contextType, s.context = typeof y == "object" && y !== null ? bl(y) : po, s.state = n.memoizedState, y = u.getDerivedStateFromProps, typeof y == "function" && (Ei(
        n,
        u,
        y,
        c
      ), s.state = n.memoizedState), typeof u.getDerivedStateFromProps == "function" || typeof s.getSnapshotBeforeUpdate == "function" || typeof s.UNSAFE_componentWillMount != "function" && typeof s.componentWillMount != "function" || (y = s.state, typeof s.componentWillMount == "function" && s.componentWillMount(), typeof s.UNSAFE_componentWillMount == "function" && s.UNSAFE_componentWillMount(), y !== s.state && rd.enqueueReplaceState(s, s.state, null), xu(n, c, s, r), Oo(), s.state = n.memoizedState), typeof s.componentDidMount == "function" && (n.flags |= 4194308), c = !0;
    } else if (l === null) {
      s = n.stateNode;
      var p = n.memoizedProps, S = Ri(u, p);
      s.props = S;
      var C = s.context, Z = u.contextType;
      y = po, typeof Z == "object" && Z !== null && (y = bl(Z));
      var W = u.getDerivedStateFromProps;
      Z = typeof W == "function" || typeof s.getSnapshotBeforeUpdate == "function", p = n.pendingProps !== p, Z || typeof s.UNSAFE_componentWillReceiveProps != "function" && typeof s.componentWillReceiveProps != "function" || (p || C !== y) && Ac(
        n,
        s,
        c,
        y
      ), Xn = !1;
      var N = n.memoizedState;
      s.state = N, xu(n, c, s, r), Oo(), C = n.memoizedState, p || N !== C || Xn ? (typeof W == "function" && (Ei(
        n,
        u,
        W,
        c
      ), C = n.memoizedState), (S = Xn || Co(
        n,
        u,
        S,
        c,
        N,
        C,
        y
      )) ? (Z || typeof s.UNSAFE_componentWillMount != "function" && typeof s.componentWillMount != "function" || (typeof s.componentWillMount == "function" && s.componentWillMount(), typeof s.UNSAFE_componentWillMount == "function" && s.UNSAFE_componentWillMount()), typeof s.componentDidMount == "function" && (n.flags |= 4194308)) : (typeof s.componentDidMount == "function" && (n.flags |= 4194308), n.memoizedProps = c, n.memoizedState = C), s.props = c, s.state = C, s.context = y, c = S) : (typeof s.componentDidMount == "function" && (n.flags |= 4194308), c = !1);
    } else {
      s = n.stateNode, Vs(l, n), y = n.memoizedProps, Z = Ri(u, y), s.props = Z, W = n.pendingProps, N = s.context, C = u.contextType, S = po, typeof C == "object" && C !== null && (S = bl(C)), p = u.getDerivedStateFromProps, (C = typeof p == "function" || typeof s.getSnapshotBeforeUpdate == "function") || typeof s.UNSAFE_componentWillReceiveProps != "function" && typeof s.componentWillReceiveProps != "function" || (y !== W || N !== S) && Ac(
        n,
        s,
        c,
        S
      ), Xn = !1, N = n.memoizedState, s.state = N, xu(n, c, s, r), Oo();
      var B = n.memoizedState;
      y !== W || N !== B || Xn || l !== null && l.dependencies !== null && Zf(l.dependencies) ? (typeof p == "function" && (Ei(
        n,
        u,
        p,
        c
      ), B = n.memoizedState), (Z = Xn || Co(
        n,
        u,
        Z,
        c,
        N,
        B,
        S
      ) || l !== null && l.dependencies !== null && Zf(l.dependencies)) ? (C || typeof s.UNSAFE_componentWillUpdate != "function" && typeof s.componentWillUpdate != "function" || (typeof s.componentWillUpdate == "function" && s.componentWillUpdate(c, B, S), typeof s.UNSAFE_componentWillUpdate == "function" && s.UNSAFE_componentWillUpdate(
        c,
        B,
        S
      )), typeof s.componentDidUpdate == "function" && (n.flags |= 4), typeof s.getSnapshotBeforeUpdate == "function" && (n.flags |= 1024)) : (typeof s.componentDidUpdate != "function" || y === l.memoizedProps && N === l.memoizedState || (n.flags |= 4), typeof s.getSnapshotBeforeUpdate != "function" || y === l.memoizedProps && N === l.memoizedState || (n.flags |= 1024), n.memoizedProps = c, n.memoizedState = B), s.props = c, s.state = B, s.context = S, c = Z) : (typeof s.componentDidUpdate != "function" || y === l.memoizedProps && N === l.memoizedState || (n.flags |= 4), typeof s.getSnapshotBeforeUpdate != "function" || y === l.memoizedProps && N === l.memoizedState || (n.flags |= 1024), c = !1);
    }
    return s = c, yr(l, n), c = (n.flags & 128) !== 0, s || c ? (s = n.stateNode, u = c && typeof u.getDerivedStateFromError != "function" ? null : s.render(), n.flags |= 1, l !== null && c ? (n.child = Rc(
      n,
      l.child,
      null,
      r
    ), n.child = Rc(
      n,
      null,
      u,
      r
    )) : Tl(l, n, u, r), n.memoizedState = s.state, l = n.child) : l = Wn(
      l,
      n,
      r
    ), l;
  }
  function yd(l, n, u, c) {
    return So(), n.flags |= 256, Tl(l, n, u, c), n.child;
  }
  var md = {
    dehydrated: null,
    treeContext: null,
    retryLane: 0,
    hydrationErrors: null
  };
  function Xy(l) {
    return { baseLanes: l, cachePool: Bs() };
  }
  function Qy(l, n, u) {
    return l = l !== null ? l.childLanes & ~u : 0, n && (l |= Ia), l;
  }
  function Zy(l, n, u) {
    var c = n.pendingProps, r = !1, s = (n.flags & 128) !== 0, y;
    if ((y = s) || (y = l !== null && l.memoizedState === null ? !1 : (Mt.current & 2) !== 0), y && (r = !0, n.flags &= -129), y = (n.flags & 32) !== 0, n.flags &= -33, l === null) {
      if (rt) {
        if (r ? wu(n) : $n(), rt) {
          var p = dt, S;
          if (S = p) {
            e: {
              for (S = p, p = Ka; S.nodeType !== 8; ) {
                if (!p) {
                  p = null;
                  break e;
                }
                if (S = En(
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
              treeContext: Mu !== null ? { id: pn, overflow: Zt } : null,
              retryLane: 536870912,
              hydrationErrors: null
            }, S = fa(
              18,
              null,
              null,
              0
            ), S.stateNode = p, S.return = n, n.child = S, el = n, dt = null, S = !0) : S = !1;
          }
          S || _u(n);
        }
        if (p = n.memoizedState, p !== null && (p = p.dehydrated, p !== null))
          return Hr(p) ? n.lanes = 32 : n.lanes = 536870912, null;
        Sn(n);
      }
      return p = c.children, c = c.fallback, r ? ($n(), r = n.mode, p = vd(
        { mode: "hidden", children: p },
        r
      ), c = Xa(
        c,
        r,
        u,
        null
      ), p.return = n, c.return = n, p.sibling = c, n.child = p, r = n.child, r.memoizedState = Xy(u), r.childLanes = Qy(
        l,
        y,
        u
      ), n.memoizedState = md, c) : (wu(n), pd(n, p));
    }
    if (S = l.memoizedState, S !== null && (p = S.dehydrated, p !== null)) {
      if (s)
        n.flags & 256 ? (wu(n), n.flags &= -257, n = Ai(
          l,
          n,
          u
        )) : n.memoizedState !== null ? ($n(), n.child = l.child, n.flags |= 128, n = null) : ($n(), r = c.fallback, p = n.mode, c = vd(
          { mode: "visible", children: c.children },
          p
        ), r = Xa(
          r,
          p,
          u,
          null
        ), r.flags |= 2, c.return = n, r.return = n, c.sibling = r, n.child = c, Rc(
          n,
          l.child,
          null,
          u
        ), c = n.child, c.memoizedState = Xy(u), c.childLanes = Qy(
          l,
          y,
          u
        ), n.memoizedState = md, n = r);
      else if (wu(n), Hr(p)) {
        if (y = p.nextSibling && p.nextSibling.dataset, y) var C = y.dgst;
        y = C, c = Error(_(419)), c.stack = "", c.digest = y, To({ value: c, source: null, stack: null }), n = Ai(
          l,
          n,
          u
        );
      } else if (dl || Eo(l, n, u, !1), y = (u & l.childLanes) !== 0, dl || y) {
        if (y = Ut, y !== null && (c = u & -u, c = (c & 42) !== 0 ? 1 : al(c), c = (c & (y.suspendedLanes | u)) !== 0 ? 0 : c, c !== 0 && c !== S.retryLane))
          throw S.retryLane = c, Yn(l, c), Ca(y, l, c), Jt;
        p.data === "$?" || Hc(), n = Ai(
          l,
          n,
          u
        );
      } else
        p.data === "$?" ? (n.flags |= 192, n.child = l.child, n = null) : (l = S.treeContext, dt = En(
          p.nextSibling
        ), el = n, rt = !0, Za = null, Ka = !1, l !== null && (Qa[ra++] = pn, Qa[ra++] = Zt, Qa[ra++] = Mu, pn = l.id, Zt = l.overflow, Mu = n), n = pd(
          n,
          c.children
        ), n.flags |= 4096);
      return n;
    }
    return r ? ($n(), r = c.fallback, p = n.mode, S = l.child, C = S.sibling, c = mn(S, {
      mode: "hidden",
      children: c.children
    }), c.subtreeFlags = S.subtreeFlags & 65011712, C !== null ? r = mn(C, r) : (r = Xa(
      r,
      p,
      u,
      null
    ), r.flags |= 2), r.return = n, c.return = n, c.sibling = r, n.child = c, c = r, r = n.child, p = l.child.memoizedState, p === null ? p = Xy(u) : (S = p.cachePool, S !== null ? (C = rl._currentValue, S = S.parent !== C ? { parent: C, pool: C } : S) : S = Bs(), p = {
      baseLanes: p.baseLanes | u,
      cachePool: S
    }), r.memoizedState = p, r.childLanes = Qy(
      l,
      y,
      u
    ), n.memoizedState = md, c) : (wu(n), u = l.child, l = u.sibling, u = mn(u, {
      mode: "visible",
      children: c.children
    }), u.return = n, u.sibling = null, l !== null && (y = n.deletions, y === null ? (n.deletions = [l], n.flags |= 16) : y.push(l)), n.child = u, n.memoizedState = null, u);
  }
  function pd(l, n) {
    return n = vd(
      { mode: "visible", children: n },
      l.mode
    ), n.return = l, l.child = n;
  }
  function vd(l, n) {
    return l = fa(22, l, null, n), l.lanes = 0, l.stateNode = {
      _visibility: 1,
      _pendingMarkers: null,
      _retryCache: null,
      _transitions: null
    }, l;
  }
  function Ai(l, n, u) {
    return Rc(n, l.child, null, u), l = pd(
      n,
      n.pendingProps.children
    ), l.flags |= 2, n.memoizedState = null, l;
  }
  function mr(l, n, u) {
    l.lanes |= n;
    var c = l.alternate;
    c !== null && (c.lanes |= n), xs(l.return, n, u);
  }
  function gd(l, n, u, c, r) {
    var s = l.memoizedState;
    s === null ? l.memoizedState = {
      isBackwards: n,
      rendering: null,
      renderingStartTime: 0,
      last: c,
      tail: u,
      tailMode: r
    } : (s.isBackwards = n, s.rendering = null, s.renderingStartTime = 0, s.last = c, s.tail = u, s.tailMode = r);
  }
  function bd(l, n, u) {
    var c = n.pendingProps, r = c.revealOrder, s = c.tail;
    if (Tl(l, n, c.children, u), c = Mt.current, (c & 2) !== 0)
      c = c & 1 | 2, n.flags |= 128;
    else {
      if (l !== null && (l.flags & 128) !== 0)
        e: for (l = n.child; l !== null; ) {
          if (l.tag === 13)
            l.memoizedState !== null && mr(l, u, n);
          else if (l.tag === 19)
            mr(l, u, n);
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
    switch (P(Mt, c), r) {
      case "forwards":
        for (u = n.child, r = null; u !== null; )
          l = u.alternate, l !== null && rr(l) === null && (r = u), u = u.sibling;
        u = r, u === null ? (r = n.child, n.child = null) : (r = u.sibling, u.sibling = null), gd(
          n,
          !1,
          r,
          u,
          s
        );
        break;
      case "backwards":
        for (u = null, r = n.child, n.child = null; r !== null; ) {
          if (l = r.alternate, l !== null && rr(l) === null) {
            n.child = r;
            break;
          }
          l = r.sibling, r.sibling = u, u = r, r = l;
        }
        gd(
          n,
          !0,
          u,
          null,
          s
        );
        break;
      case "together":
        gd(n, !1, null, null, void 0);
        break;
      default:
        n.memoizedState = null;
    }
    return n.child;
  }
  function Wn(l, n, u) {
    if (l !== null && (n.dependencies = l.dependencies), ju |= n.lanes, (u & n.childLanes) === 0)
      if (l !== null) {
        if (Eo(
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
      for (l = n.child, u = mn(l, l.pendingProps), n.child = u, u.return = n; l.sibling !== null; )
        l = l.sibling, u = u.sibling = mn(l, l.pendingProps), u.return = n;
      u.sibling = null;
    }
    return n.child;
  }
  function Sd(l, n) {
    return (l.lanes & n) !== 0 ? !0 : (l = l.dependencies, !!(l !== null && Zf(l)));
  }
  function ov(l, n, u) {
    switch (n.tag) {
      case 3:
        xe(n, n.stateNode.containerInfo), Cu(n, rl, l.memoizedState.cache), So();
        break;
      case 27:
      case 5:
        ua(n);
        break;
      case 4:
        xe(n, n.stateNode.containerInfo);
        break;
      case 10:
        Cu(
          n,
          n.type,
          n.memoizedProps.value
        );
        break;
      case 13:
        var c = n.memoizedState;
        if (c !== null)
          return c.dehydrated !== null ? (wu(n), n.flags |= 128, null) : (u & n.child.childLanes) !== 0 ? Zy(l, n, u) : (wu(n), l = Wn(
            l,
            n,
            u
          ), l !== null ? l.sibling : null);
        wu(n);
        break;
      case 19:
        var r = (l.flags & 128) !== 0;
        if (c = (u & n.childLanes) !== 0, c || (Eo(
          l,
          n,
          u,
          !1
        ), c = (u & n.childLanes) !== 0), r) {
          if (c)
            return bd(
              l,
              n,
              u
            );
          n.flags |= 128;
        }
        if (r = n.memoizedState, r !== null && (r.rendering = null, r.tail = null, r.lastEffect = null), P(Mt, Mt.current), c) break;
        return null;
      case 22:
      case 23:
        return n.lanes = 0, sd(l, n, u);
      case 24:
        Cu(n, rl, l.memoizedState.cache);
    }
    return Wn(l, n, u);
  }
  function fv(l, n, u) {
    if (l !== null)
      if (l.memoizedProps !== n.pendingProps)
        dl = !0;
      else {
        if (!Sd(l, u) && (n.flags & 128) === 0)
          return dl = !1, ov(
            l,
            n,
            u
          );
        dl = (l.flags & 131072) !== 0;
      }
    else
      dl = !1, rt && (n.flags & 1048576) !== 0 && Cs(n, go, n.index);
    switch (n.lanes = 0, n.tag) {
      case 16:
        e: {
          l = n.pendingProps;
          var c = n.elementType, r = c._init;
          if (c = r(c._payload), n.type = c, typeof c == "function")
            Lf(c) ? (l = Ri(c, l), n.tag = 1, n = hd(
              null,
              n,
              c,
              l,
              u
            )) : (n.tag = 0, n = dd(
              null,
              n,
              c,
              l,
              u
            ));
          else {
            if (c != null) {
              if (r = c.$$typeof, r === At) {
                n.tag = 11, n = cv(
                  null,
                  n,
                  c,
                  l,
                  u
                );
                break e;
              } else if (r === je) {
                n.tag = 14, n = qu(
                  null,
                  n,
                  c,
                  l,
                  u
                );
                break e;
              }
            }
            throw n = Ht(c) || c, Error(_(306, n, ""));
          }
        }
        return n;
      case 0:
        return dd(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 1:
        return c = n.type, r = Ri(
          c,
          n.pendingProps
        ), hd(
          l,
          n,
          c,
          r,
          u
        );
      case 3:
        e: {
          if (xe(
            n,
            n.stateNode.containerInfo
          ), l === null) throw Error(_(387));
          c = n.pendingProps;
          var s = n.memoizedState;
          r = s.element, Vs(l, n), xu(n, c, null, u);
          var y = n.memoizedState;
          if (c = y.cache, Cu(n, rl, c), c !== s.cache && hy(
            n,
            [rl],
            u,
            !0
          ), Oo(), c = y.element, s.isDehydrated)
            if (s = {
              element: c,
              isDehydrated: !1,
              cache: y.cache
            }, n.updateQueue.baseState = s, n.memoizedState = s, n.flags & 256) {
              n = yd(
                l,
                n,
                c,
                u
              );
              break e;
            } else if (c !== r) {
              r = Da(
                Error(_(424)),
                n
              ), To(r), n = yd(
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
              for (dt = En(l.firstChild), el = n, rt = !0, Za = null, Ka = !0, u = Jn(
                n,
                null,
                c,
                u
              ), n.child = u; u; )
                u.flags = u.flags & -3 | 4096, u = u.sibling;
            }
          else {
            if (So(), c === r) {
              n = Wn(
                l,
                n,
                u
              );
              break e;
            }
            Tl(
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
        return yr(l, n), l === null ? (u = Ov(
          n.type,
          null,
          n.pendingProps,
          null
        )) ? n.memoizedState = u : rt || (u = n.type, l = n.pendingProps, c = en(
          oe.current
        ).createElement(u), c[gl] = n, c[$l] = l, Ue(c, u, l), fl(c), n.stateNode = c) : n.memoizedState = Ov(
          n.type,
          l.memoizedProps,
          n.pendingProps,
          l.memoizedState
        ), null;
      case 27:
        return ua(n), l === null && rt && (c = n.stateNode = fe(
          n.type,
          n.pendingProps,
          oe.current
        ), el = n, Ka = !0, r = dt, Hi(n.type) ? (Ni = r, dt = En(
          c.firstChild
        )) : dt = r), Tl(
          l,
          n,
          n.pendingProps.children,
          u
        ), yr(l, n), l === null && (n.flags |= 4194304), n.child;
      case 5:
        return l === null && rt && ((r = c = dt) && (c = Fo(
          c,
          n.type,
          n.pendingProps,
          Ka
        ), c !== null ? (n.stateNode = c, el = n, dt = En(
          c.firstChild
        ), Ka = !1, r = !0) : r = !1), r || _u(n)), ua(n), r = n.type, s = n.pendingProps, y = l !== null ? l.memoizedProps : null, c = s.children, uu(r, s) ? c = null : y !== null && uu(r, y) && (n.flags |= 32), n.memoizedState !== null && (r = Zs(
          l,
          n,
          Fp,
          null,
          null,
          u
        ), Sa._currentValue = r), yr(l, n), Tl(l, n, c, u), n.child;
      case 6:
        return l === null && rt && ((l = u = dt) && (u = jg(
          u,
          n.pendingProps,
          Ka
        ), u !== null ? (n.stateNode = u, el = n, dt = null, l = !0) : l = !1), l || _u(n)), null;
      case 13:
        return Zy(l, n, u);
      case 4:
        return xe(
          n,
          n.stateNode.containerInfo
        ), c = n.pendingProps, l === null ? n.child = Rc(
          n,
          null,
          c,
          u
        ) : Tl(
          l,
          n,
          c,
          u
        ), n.child;
      case 11:
        return cv(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 7:
        return Tl(
          l,
          n,
          n.pendingProps,
          u
        ), n.child;
      case 8:
        return Tl(
          l,
          n,
          n.pendingProps.children,
          u
        ), n.child;
      case 12:
        return Tl(
          l,
          n,
          n.pendingProps.children,
          u
        ), n.child;
      case 10:
        return c = n.pendingProps, Cu(n, n.type, c.value), Tl(
          l,
          n,
          c.children,
          u
        ), n.child;
      case 9:
        return r = n.type._context, c = n.pendingProps.children, pi(n), r = bl(r), c = c(r), n.flags |= 1, Tl(l, n, c, u), n.child;
      case 14:
        return qu(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 15:
        return Oc(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 19:
        return bd(l, n, u);
      case 31:
        return c = n.pendingProps, u = n.mode, c = {
          mode: c.mode,
          children: c.children
        }, l === null ? (u = vd(
          c,
          u
        ), u.ref = n.ref, n.child = u, u.return = n, n = u) : (u = mn(l.child, c), u.ref = n.ref, n.child = u, u.return = n, n = u), n;
      case 22:
        return sd(l, n, u);
      case 24:
        return pi(n), c = bl(rl), l === null ? (r = kf(), r === null && (r = Ut, s = Ao(), r.pooledCache = s, s.refCount++, s !== null && (r.pooledCacheLanes |= u), r = s), n.memoizedState = {
          parent: c,
          cache: r
        }, Ls(n), Cu(n, rl, r)) : ((l.lanes & u) !== 0 && (Vs(l, n), xu(n, null, null, u), Oo()), r = l.memoizedState, s = n.memoizedState, r.parent !== c ? (r = { parent: c, cache: c }, n.memoizedState = r, n.lanes === 0 && (n.memoizedState = n.updateQueue.baseState = r), Cu(n, rl, c)) : (c = s.cache, Cu(n, rl, c), c !== r.cache && hy(
          n,
          [rl],
          u,
          !0
        ))), Tl(
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
  function Fn(l) {
    l.flags |= 4;
  }
  function Ho(l, n) {
    if (n.type !== "stylesheet" || (n.state.loading & 4) !== 0)
      l.flags &= -16777217;
    else if (l.flags |= 16777216, !Mm(n)) {
      if (n = _a.current, n !== null && ((lt & 4194048) === lt ? Ql !== null : (lt & 62914560) !== lt && (lt & 536870912) === 0 || n !== Ql))
        throw yc = js, Ys;
      l.flags |= 8192;
    }
  }
  function pr(l, n) {
    n !== null && (l.flags |= 4), l.flags & 16384 && (n = l.tag !== 22 ? ne() : 536870912, l.lanes |= n, Yo |= n);
  }
  function No(l, n) {
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
      for (var r = l.child; r !== null; )
        u |= r.lanes | r.childLanes, c |= r.subtreeFlags & 65011712, c |= r.flags & 65011712, r.return = l, r = r.sibling;
    else
      for (r = l.child; r !== null; )
        u |= r.lanes | r.childLanes, c |= r.subtreeFlags, c |= r.flags, r.return = l, r = r.sibling;
    return l.subtreeFlags |= c, l.childLanes = u, n;
  }
  function Ky(l, n, u) {
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
        return u = n.stateNode, c = null, l !== null && (c = l.memoizedState.cache), n.memoizedState.cache !== c && (n.flags |= 2048), Gn(rl), Bt(), u.pendingContext && (u.context = u.pendingContext, u.pendingContext = null), (l === null || l.child === null) && (bo(n) ? Fn(n) : l === null || l.memoizedState.isDehydrated && (n.flags & 256) === 0 || (n.flags |= 1024, dy())), Me(n), null;
      case 26:
        return u = n.memoizedState, l === null ? (Fn(n), u !== null ? (Me(n), Ho(n, u)) : (Me(n), n.flags &= -16777217)) : u ? u !== l.memoizedState ? (Fn(n), Me(n), Ho(n, u)) : (Me(n), n.flags &= -16777217) : (l.memoizedProps !== c && Fn(n), Me(n), n.flags &= -16777217), null;
      case 27:
        Mn(n), u = oe.current;
        var r = n.type;
        if (l !== null && n.stateNode != null)
          l.memoizedProps !== c && Fn(n);
        else {
          if (!c) {
            if (n.stateNode === null)
              throw Error(_(166));
            return Me(n), null;
          }
          l = ce.current, bo(n) ? Xf(n) : (l = fe(r, c, u), n.stateNode = l, Fn(n));
        }
        return Me(n), null;
      case 5:
        if (Mn(n), u = n.type, l !== null && n.stateNode != null)
          l.memoizedProps !== c && Fn(n);
        else {
          if (!c) {
            if (n.stateNode === null)
              throw Error(_(166));
            return Me(n), null;
          }
          if (l = ce.current, bo(n))
            Xf(n);
          else {
            switch (r = en(
              oe.current
            ), l) {
              case 1:
                l = r.createElementNS(
                  "http://www.w3.org/2000/svg",
                  u
                );
                break;
              case 2:
                l = r.createElementNS(
                  "http://www.w3.org/1998/Math/MathML",
                  u
                );
                break;
              default:
                switch (u) {
                  case "svg":
                    l = r.createElementNS(
                      "http://www.w3.org/2000/svg",
                      u
                    );
                    break;
                  case "math":
                    l = r.createElementNS(
                      "http://www.w3.org/1998/Math/MathML",
                      u
                    );
                    break;
                  case "script":
                    l = r.createElement("div"), l.innerHTML = "<script><\/script>", l = l.removeChild(l.firstChild);
                    break;
                  case "select":
                    l = typeof c.is == "string" ? r.createElement("select", { is: c.is }) : r.createElement("select"), c.multiple ? l.multiple = !0 : c.size && (l.size = c.size);
                    break;
                  default:
                    l = typeof c.is == "string" ? r.createElement(u, { is: c.is }) : r.createElement(u);
                }
            }
            l[gl] = n, l[$l] = c;
            e: for (r = n.child; r !== null; ) {
              if (r.tag === 5 || r.tag === 6)
                l.appendChild(r.stateNode);
              else if (r.tag !== 4 && r.tag !== 27 && r.child !== null) {
                r.child.return = r, r = r.child;
                continue;
              }
              if (r === n) break e;
              for (; r.sibling === null; ) {
                if (r.return === null || r.return === n)
                  break e;
                r = r.return;
              }
              r.sibling.return = r.return, r = r.sibling;
            }
            n.stateNode = l;
            e: switch (Ue(l, u, c), u) {
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
            l && Fn(n);
          }
        }
        return Me(n), n.flags &= -16777217, null;
      case 6:
        if (l && n.stateNode != null)
          l.memoizedProps !== c && Fn(n);
        else {
          if (typeof c != "string" && n.stateNode === null)
            throw Error(_(166));
          if (l = oe.current, bo(n)) {
            if (l = n.stateNode, u = n.memoizedProps, c = null, r = el, r !== null)
              switch (r.tag) {
                case 27:
                case 5:
                  c = r.memoizedProps;
              }
            l[gl] = n, l = !!(l.nodeValue === u || c !== null && c.suppressHydrationWarning === !0 || Am(l.nodeValue, u)), l || _u(n);
          } else
            l = en(l).createTextNode(
              c
            ), l[gl] = n, n.stateNode = l;
        }
        return Me(n), null;
      case 13:
        if (c = n.memoizedState, l === null || l.memoizedState !== null && l.memoizedState.dehydrated !== null) {
          if (r = bo(n), c !== null && c.dehydrated !== null) {
            if (l === null) {
              if (!r) throw Error(_(318));
              if (r = n.memoizedState, r = r !== null ? r.dehydrated : null, !r) throw Error(_(317));
              r[gl] = n;
            } else
              So(), (n.flags & 128) === 0 && (n.memoizedState = null), n.flags |= 4;
            Me(n), r = !1;
          } else
            r = dy(), l !== null && l.memoizedState !== null && (l.memoizedState.hydrationErrors = r), r = !0;
          if (!r)
            return n.flags & 256 ? (Sn(n), n) : (Sn(n), null);
        }
        if (Sn(n), (n.flags & 128) !== 0)
          return n.lanes = u, n;
        if (u = c !== null, l = l !== null && l.memoizedState !== null, u) {
          c = n.child, r = null, c.alternate !== null && c.alternate.memoizedState !== null && c.alternate.memoizedState.cachePool !== null && (r = c.alternate.memoizedState.cachePool.pool);
          var s = null;
          c.memoizedState !== null && c.memoizedState.cachePool !== null && (s = c.memoizedState.cachePool.pool), s !== r && (c.flags |= 2048);
        }
        return u !== l && u && (n.child.flags |= 8192), pr(n, n.updateQueue), Me(n), null;
      case 4:
        return Bt(), l === null && Em(n.stateNode.containerInfo), Me(n), null;
      case 10:
        return Gn(n.type), Me(n), null;
      case 19:
        if (J(Mt), r = n.memoizedState, r === null) return Me(n), null;
        if (c = (n.flags & 128) !== 0, s = r.rendering, s === null)
          if (c) No(r, !1);
          else {
            if (Wt !== 0 || l !== null && (l.flags & 128) !== 0)
              for (l = n.child; l !== null; ) {
                if (s = rr(l), s !== null) {
                  for (n.flags |= 128, No(r, !1), l = s.updateQueue, n.updateQueue = l, pr(n, l), n.subtreeFlags = 0, l = u, u = n.child; u !== null; )
                    $e(u, l), u = u.sibling;
                  return P(
                    Mt,
                    Mt.current & 1 | 2
                  ), n.child;
                }
                l = l.sibling;
              }
            r.tail !== null && vl() > _d && (n.flags |= 128, c = !0, No(r, !1), n.lanes = 4194304);
          }
        else {
          if (!c)
            if (l = rr(s), l !== null) {
              if (n.flags |= 128, c = !0, l = l.updateQueue, n.updateQueue = l, pr(n, l), No(r, !0), r.tail === null && r.tailMode === "hidden" && !s.alternate && !rt)
                return Me(n), null;
            } else
              2 * vl() - r.renderingStartTime > _d && u !== 536870912 && (n.flags |= 128, c = !0, No(r, !1), n.lanes = 4194304);
          r.isBackwards ? (s.sibling = n.child, n.child = s) : (l = r.last, l !== null ? l.sibling = s : n.child = s, r.last = s);
        }
        return r.tail !== null ? (n = r.tail, r.rendering = n, r.tail = n.sibling, r.renderingStartTime = vl(), n.sibling = null, l = Mt.current, P(Mt, c ? l & 1 | 2 : l & 1), n) : (Me(n), null);
      case 22:
      case 23:
        return Sn(n), zo(), c = n.memoizedState !== null, l !== null ? l.memoizedState !== null !== c && (n.flags |= 8192) : c && (n.flags |= 8192), c ? (u & 536870912) !== 0 && (n.flags & 128) === 0 && (Me(n), n.subtreeFlags & 6 && (n.flags |= 8192)) : Me(n), u = n.updateQueue, u !== null && pr(n, u.retryQueue), u = null, l !== null && l.memoizedState !== null && l.memoizedState.cachePool !== null && (u = l.memoizedState.cachePool.pool), c = null, n.memoizedState !== null && n.memoizedState.cachePool !== null && (c = n.memoizedState.cachePool.pool), c !== u && (n.flags |= 2048), l !== null && J(Vn), null;
      case 24:
        return u = null, l !== null && (u = l.memoizedState.cache), n.memoizedState.cache !== u && (n.flags |= 2048), Gn(rl), Me(n), null;
      case 25:
        return null;
      case 30:
        return null;
    }
    throw Error(_(156, n.tag));
  }
  function Cg(l, n) {
    switch (jn(n), n.tag) {
      case 1:
        return l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 3:
        return Gn(rl), Bt(), l = n.flags, (l & 65536) !== 0 && (l & 128) === 0 ? (n.flags = l & -65537 | 128, n) : null;
      case 26:
      case 27:
      case 5:
        return Mn(n), null;
      case 13:
        if (Sn(n), l = n.memoizedState, l !== null && l.dehydrated !== null) {
          if (n.alternate === null)
            throw Error(_(340));
          So();
        }
        return l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 19:
        return J(Mt), null;
      case 4:
        return Bt(), null;
      case 10:
        return Gn(n.type), null;
      case 22:
      case 23:
        return Sn(n), zo(), l !== null && J(Vn), l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 24:
        return Gn(rl), null;
      case 25:
        return null;
      default:
        return null;
    }
  }
  function Jy(l, n) {
    switch (jn(n), n.tag) {
      case 3:
        Gn(rl), Bt();
        break;
      case 26:
      case 27:
      case 5:
        Mn(n);
        break;
      case 4:
        Bt();
        break;
      case 13:
        Sn(n);
        break;
      case 19:
        J(Mt);
        break;
      case 10:
        Gn(n.type);
        break;
      case 22:
      case 23:
        Sn(n), zo(), l !== null && J(Vn);
        break;
      case 24:
        Gn(rl);
    }
  }
  function vr(l, n) {
    try {
      var u = n.updateQueue, c = u !== null ? u.lastEffect : null;
      if (c !== null) {
        var r = c.next;
        u = r;
        do {
          if ((u.tag & l) === l) {
            c = void 0;
            var s = u.create, y = u.inst;
            c = s(), y.destroy = c;
          }
          u = u.next;
        } while (u !== r);
      }
    } catch (p) {
      Tt(n, n.return, p);
    }
  }
  function Oi(l, n, u) {
    try {
      var c = n.updateQueue, r = c !== null ? c.lastEffect : null;
      if (r !== null) {
        var s = r.next;
        c = s;
        do {
          if ((c.tag & l) === l) {
            var y = c.inst, p = y.destroy;
            if (p !== void 0) {
              y.destroy = void 0, r = n;
              var S = u, C = p;
              try {
                C();
              } catch (Z) {
                Tt(
                  r,
                  S,
                  Z
                );
              }
            }
          }
          c = c.next;
        } while (c !== s);
      }
    } catch (Z) {
      Tt(n, n.return, Z);
    }
  }
  function Td(l) {
    var n = l.updateQueue;
    if (n !== null) {
      var u = l.stateNode;
      try {
        Ff(n, u);
      } catch (c) {
        Tt(l, l.return, c);
      }
    }
  }
  function ky(l, n, u) {
    u.props = Ri(
      l.type,
      l.memoizedProps
    ), u.state = l.memoizedState;
    try {
      u.componentWillUnmount();
    } catch (c) {
      Tt(l, n, c);
    }
  }
  function wo(l, n) {
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
    } catch (r) {
      Tt(l, n, r);
    }
  }
  function Tn(l, n) {
    var u = l.ref, c = l.refCleanup;
    if (u !== null)
      if (typeof c == "function")
        try {
          c();
        } catch (r) {
          Tt(l, n, r);
        } finally {
          l.refCleanup = null, l = l.alternate, l != null && (l.refCleanup = null);
        }
      else if (typeof u == "function")
        try {
          u(null);
        } catch (r) {
          Tt(l, n, r);
        }
      else u.current = null;
  }
  function qo(l) {
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
    } catch (r) {
      Tt(l, l.return, r);
    }
  }
  function $y(l, n, u) {
    try {
      var c = l.stateNode;
      qg(c, l.type, u, n), c[$l] = n;
    } catch (r) {
      Tt(l, l.return, r);
    }
  }
  function rv(l) {
    return l.tag === 5 || l.tag === 3 || l.tag === 26 || l.tag === 27 && Hi(l.type) || l.tag === 4;
  }
  function Wa(l) {
    e: for (; ; ) {
      for (; l.sibling === null; ) {
        if (l.return === null || rv(l.return)) return null;
        l = l.return;
      }
      for (l.sibling.return = l.return, l = l.sibling; l.tag !== 5 && l.tag !== 6 && l.tag !== 18; ) {
        if (l.tag === 27 && Hi(l.type) || l.flags & 2 || l.child === null || l.tag === 4) continue e;
        l.child.return = l, l = l.child;
      }
      if (!(l.flags & 2)) return l.stateNode;
    }
  }
  function zc(l, n, u) {
    var c = l.tag;
    if (c === 5 || c === 6)
      l = l.stateNode, n ? (u.nodeType === 9 ? u.body : u.nodeName === "HTML" ? u.ownerDocument.body : u).insertBefore(l, n) : (n = u.nodeType === 9 ? u.body : u.nodeName === "HTML" ? u.ownerDocument.body : u, n.appendChild(l), u = u._reactRootContainer, u != null || n.onclick !== null || (n.onclick = Ld));
    else if (c !== 4 && (c === 27 && Hi(l.type) && (u = l.stateNode, n = null), l = l.child, l !== null))
      for (zc(l, n, u), l = l.sibling; l !== null; )
        zc(l, n, u), l = l.sibling;
  }
  function Ed(l, n, u) {
    var c = l.tag;
    if (c === 5 || c === 6)
      l = l.stateNode, n ? u.insertBefore(l, n) : u.appendChild(l);
    else if (c !== 4 && (c === 27 && Hi(l.type) && (u = l.stateNode), l = l.child, l !== null))
      for (Ed(l, n, u), l = l.sibling; l !== null; )
        Ed(l, n, u), l = l.sibling;
  }
  function Rd(l) {
    var n = l.stateNode, u = l.memoizedProps;
    try {
      for (var c = l.type, r = n.attributes; r.length; )
        n.removeAttributeNode(r[0]);
      Ue(n, c, u), n[gl] = l, n[$l] = u;
    } catch (s) {
      Tt(l, l.return, s);
    }
  }
  var In = !1, kt = !1, Ad = !1, Od = typeof WeakSet == "function" ? WeakSet : Set, hl = null;
  function Wy(l, n) {
    if (l = l.containerInfo, Ur = qr, l = uy(l), jf(l)) {
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
            var r = c.anchorOffset, s = c.focusNode;
            c = c.focusOffset;
            try {
              u.nodeType, s.nodeType;
            } catch {
              u = null;
              break e;
            }
            var y = 0, p = -1, S = -1, C = 0, Z = 0, W = l, N = null;
            t: for (; ; ) {
              for (var B; W !== u || r !== 0 && W.nodeType !== 3 || (p = y + r), W !== s || c !== 0 && W.nodeType !== 3 || (S = y + c), W.nodeType === 3 && (y += W.nodeValue.length), (B = W.firstChild) !== null; )
                N = W, W = B;
              for (; ; ) {
                if (W === l) break t;
                if (N === u && ++C === r && (p = y), N === s && ++Z === c && (S = y), (B = W.nextSibling) !== null) break;
                W = N, N = W.parentNode;
              }
              W = B;
            }
            u = p === -1 || S === -1 ? null : { start: p, end: S };
          } else u = null;
        }
      u = u || { start: 0, end: 0 };
    } else u = null;
    for (Cr = { focusedElem: l, selectionRange: u }, qr = !1, hl = n; hl !== null; )
      if (n = hl, l = n.child, (n.subtreeFlags & 1024) !== 0 && l !== null)
        l.return = n, hl = l;
      else
        for (; hl !== null; ) {
          switch (n = hl, s = n.alternate, l = n.flags, n.tag) {
            case 0:
              break;
            case 11:
            case 15:
              break;
            case 1:
              if ((l & 1024) !== 0 && s !== null) {
                l = void 0, u = n, r = s.memoizedProps, s = s.memoizedState, c = u.stateNode;
                try {
                  var Te = Ri(
                    u.type,
                    r,
                    u.elementType === u.type
                  );
                  l = c.getSnapshotBeforeUpdate(
                    Te,
                    s
                  ), c.__reactInternalSnapshotBeforeUpdate = l;
                } catch (Ee) {
                  Tt(
                    u,
                    u.return,
                    Ee
                  );
                }
              }
              break;
            case 3:
              if ((l & 1024) !== 0) {
                if (l = n.stateNode.containerInfo, u = l.nodeType, u === 9)
                  xr(l);
                else if (u === 1)
                  switch (l.nodeName) {
                    case "HEAD":
                    case "HTML":
                    case "BODY":
                      xr(l);
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
            l.return = n.return, hl = l;
            break;
          }
          hl = n.return;
        }
  }
  function Fy(l, n, u) {
    var c = u.flags;
    switch (u.tag) {
      case 0:
      case 11:
      case 15:
        eu(l, u), c & 4 && vr(5, u);
        break;
      case 1:
        if (eu(l, u), c & 4)
          if (l = u.stateNode, n === null)
            try {
              l.componentDidMount();
            } catch (y) {
              Tt(u, u.return, y);
            }
          else {
            var r = Ri(
              u.type,
              n.memoizedProps
            );
            n = n.memoizedState;
            try {
              l.componentDidUpdate(
                r,
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
        c & 64 && Td(u), c & 512 && wo(u, u.return);
        break;
      case 3:
        if (eu(l, u), c & 64 && (l = u.updateQueue, l !== null)) {
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
            Ff(l, n);
          } catch (y) {
            Tt(u, u.return, y);
          }
        }
        break;
      case 27:
        n === null && c & 4 && Rd(u);
      case 26:
      case 5:
        eu(l, u), n === null && c & 4 && qo(u), c & 512 && wo(u, u.return);
        break;
      case 12:
        eu(l, u);
        break;
      case 13:
        eu(l, u), c & 4 && Dd(l, u), c & 64 && (l = u.memoizedState, l !== null && (l = l.dehydrated, l !== null && (u = xg.bind(
          null,
          u
        ), Gg(l, u))));
        break;
      case 22:
        if (c = u.memoizedState !== null || In, !c) {
          n = n !== null && n.memoizedState !== null || kt, r = In;
          var s = kt;
          In = c, (kt = n) && !s ? Di(
            l,
            u,
            (u.subtreeFlags & 8772) !== 0
          ) : eu(l, u), In = r, kt = s;
        }
        break;
      case 30:
        break;
      default:
        eu(l, u);
    }
  }
  function Iy(l) {
    var n = l.alternate;
    n !== null && (l.alternate = null, Iy(n)), l.child = null, l.deletions = null, l.sibling = null, l.tag === 5 && (n = l.stateNode, n !== null && Rf(n)), l.stateNode = null, l.return = null, l.dependencies = null, l.memoizedProps = null, l.memoizedState = null, l.pendingProps = null, l.stateNode = null, l.updateQueue = null;
  }
  var Yt = null, xl = !1;
  function Pn(l, n, u) {
    for (u = u.child; u !== null; )
      et(l, n, u), u = u.sibling;
  }
  function et(l, n, u) {
    if (Dl && typeof Dl.onCommitFiberUnmount == "function")
      try {
        Dl.onCommitFiberUnmount(ei, u);
      } catch {
      }
    switch (u.tag) {
      case 26:
        kt || Tn(u, n), Pn(
          l,
          n,
          u
        ), u.memoizedState ? u.memoizedState.count-- : u.stateNode && (u = u.stateNode, u.parentNode.removeChild(u));
        break;
      case 27:
        kt || Tn(u, n);
        var c = Yt, r = xl;
        Hi(u.type) && (Yt = u.stateNode, xl = !1), Pn(
          l,
          n,
          u
        ), ga(u.stateNode), Yt = c, xl = r;
        break;
      case 5:
        kt || Tn(u, n);
      case 6:
        if (c = Yt, r = xl, Yt = null, Pn(
          l,
          n,
          u
        ), Yt = c, xl = r, Yt !== null)
          if (xl)
            try {
              (Yt.nodeType === 9 ? Yt.body : Yt.nodeName === "HTML" ? Yt.ownerDocument.body : Yt).removeChild(u.stateNode);
            } catch (s) {
              Tt(
                u,
                n,
                s
              );
            }
          else
            try {
              Yt.removeChild(u.stateNode);
            } catch (s) {
              Tt(
                u,
                n,
                s
              );
            }
        break;
      case 18:
        Yt !== null && (xl ? (l = Yt, Xd(
          l.nodeType === 9 ? l.body : l.nodeName === "HTML" ? l.ownerDocument.body : l,
          u.stateNode
        ), ou(l)) : Xd(Yt, u.stateNode));
        break;
      case 4:
        c = Yt, r = xl, Yt = u.stateNode.containerInfo, xl = !0, Pn(
          l,
          n,
          u
        ), Yt = c, xl = r;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        kt || Oi(2, u, n), kt || Oi(4, u, n), Pn(
          l,
          n,
          u
        );
        break;
      case 1:
        kt || (Tn(u, n), c = u.stateNode, typeof c.componentWillUnmount == "function" && ky(
          u,
          n,
          c
        )), Pn(
          l,
          n,
          u
        );
        break;
      case 21:
        Pn(
          l,
          n,
          u
        );
        break;
      case 22:
        kt = (c = kt) || u.memoizedState !== null, Pn(
          l,
          n,
          u
        ), kt = c;
        break;
      default:
        Pn(
          l,
          n,
          u
        );
    }
  }
  function Dd(l, n) {
    if (n.memoizedState === null && (l = n.alternate, l !== null && (l = l.memoizedState, l !== null && (l = l.dehydrated, l !== null))))
      try {
        ou(l);
      } catch (u) {
        Tt(n, n.return, u);
      }
  }
  function Py(l) {
    switch (l.tag) {
      case 13:
      case 19:
        var n = l.stateNode;
        return n === null && (n = l.stateNode = new Od()), n;
      case 22:
        return l = l.stateNode, n = l._retryCache, n === null && (n = l._retryCache = new Od()), n;
      default:
        throw Error(_(435, l.tag));
    }
  }
  function zd(l, n) {
    var u = Py(l);
    n.forEach(function(c) {
      var r = Hg.bind(null, l, c);
      u.has(c) || (u.add(c), c.then(r, r));
    });
  }
  function Il(l, n) {
    var u = n.deletions;
    if (u !== null)
      for (var c = 0; c < u.length; c++) {
        var r = u[c], s = l, y = n, p = y;
        e: for (; p !== null; ) {
          switch (p.tag) {
            case 27:
              if (Hi(p.type)) {
                Yt = p.stateNode, xl = !1;
                break e;
              }
              break;
            case 5:
              Yt = p.stateNode, xl = !1;
              break e;
            case 3:
            case 4:
              Yt = p.stateNode.containerInfo, xl = !0;
              break e;
          }
          p = p.return;
        }
        if (Yt === null) throw Error(_(160));
        et(s, y, r), Yt = null, xl = !1, s = r.alternate, s !== null && (s.return = null), r.return = null;
      }
    if (n.subtreeFlags & 13878)
      for (n = n.child; n !== null; )
        gr(n, l), n = n.sibling;
  }
  var Pl = null;
  function gr(l, n) {
    var u = l.alternate, c = l.flags;
    switch (l.tag) {
      case 0:
      case 11:
      case 14:
      case 15:
        Il(n, l), El(l), c & 4 && (Oi(3, l, l.return), vr(3, l), Oi(5, l, l.return));
        break;
      case 1:
        Il(n, l), El(l), c & 512 && (kt || u === null || Tn(u, u.return)), c & 64 && In && (l = l.updateQueue, l !== null && (c = l.callbacks, c !== null && (u = l.shared.hiddenCallbacks, l.shared.hiddenCallbacks = u === null ? c : u.concat(c))));
        break;
      case 26:
        var r = Pl;
        if (Il(n, l), El(l), c & 512 && (kt || u === null || Tn(u, u.return)), c & 4) {
          var s = u !== null ? u.memoizedState : null;
          if (c = l.memoizedState, u === null)
            if (c === null)
              if (l.stateNode === null) {
                e: {
                  c = l.type, u = l.memoizedProps, r = r.ownerDocument || r;
                  t: switch (c) {
                    case "title":
                      s = r.getElementsByTagName("title")[0], (!s || s[he] || s[gl] || s.namespaceURI === "http://www.w3.org/2000/svg" || s.hasAttribute("itemprop")) && (s = r.createElement(c), r.head.insertBefore(
                        s,
                        r.querySelector("head > title")
                      )), Ue(s, c, u), s[gl] = l, fl(s), c = s;
                      break e;
                    case "link":
                      var y = Dm(
                        "link",
                        "href",
                        r
                      ).get(c + (u.href || ""));
                      if (y) {
                        for (var p = 0; p < y.length; p++)
                          if (s = y[p], s.getAttribute("href") === (u.href == null || u.href === "" ? null : u.href) && s.getAttribute("rel") === (u.rel == null ? null : u.rel) && s.getAttribute("title") === (u.title == null ? null : u.title) && s.getAttribute("crossorigin") === (u.crossOrigin == null ? null : u.crossOrigin)) {
                            y.splice(p, 1);
                            break t;
                          }
                      }
                      s = r.createElement(c), Ue(s, c, u), r.head.appendChild(s);
                      break;
                    case "meta":
                      if (y = Dm(
                        "meta",
                        "content",
                        r
                      ).get(c + (u.content || ""))) {
                        for (p = 0; p < y.length; p++)
                          if (s = y[p], s.getAttribute("content") === (u.content == null ? null : "" + u.content) && s.getAttribute("name") === (u.name == null ? null : u.name) && s.getAttribute("property") === (u.property == null ? null : u.property) && s.getAttribute("http-equiv") === (u.httpEquiv == null ? null : u.httpEquiv) && s.getAttribute("charset") === (u.charSet == null ? null : u.charSet)) {
                            y.splice(p, 1);
                            break t;
                          }
                      }
                      s = r.createElement(c), Ue(s, c, u), r.head.appendChild(s);
                      break;
                    default:
                      throw Error(_(468, c));
                  }
                  s[gl] = l, fl(s), c = s;
                }
                l.stateNode = c;
              } else
                zm(
                  r,
                  l.type,
                  l.stateNode
                );
            else
              l.stateNode = zv(
                r,
                c,
                l.memoizedProps
              );
          else
            s !== c ? (s === null ? u.stateNode !== null && (u = u.stateNode, u.parentNode.removeChild(u)) : s.count--, c === null ? zm(
              r,
              l.type,
              l.stateNode
            ) : zv(
              r,
              c,
              l.memoizedProps
            )) : c === null && l.stateNode !== null && $y(
              l,
              l.memoizedProps,
              u.memoizedProps
            );
        }
        break;
      case 27:
        Il(n, l), El(l), c & 512 && (kt || u === null || Tn(u, u.return)), u !== null && c & 4 && $y(
          l,
          l.memoizedProps,
          u.memoizedProps
        );
        break;
      case 5:
        if (Il(n, l), El(l), c & 512 && (kt || u === null || Tn(u, u.return)), l.flags & 32) {
          r = l.stateNode;
          try {
            io(r, "");
          } catch (B) {
            Tt(l, l.return, B);
          }
        }
        c & 4 && l.stateNode != null && (r = l.memoizedProps, $y(
          l,
          r,
          u !== null ? u.memoizedProps : r
        )), c & 1024 && (Ad = !0);
        break;
      case 6:
        if (Il(n, l), El(l), c & 4) {
          if (l.stateNode === null)
            throw Error(_(162));
          c = l.memoizedProps, u = l.stateNode;
          try {
            u.nodeValue = c;
          } catch (B) {
            Tt(l, l.return, B);
          }
        }
        break;
      case 3:
        if (Bi = null, r = Pl, Pl = Qd(n.containerInfo), Il(n, l), Pl = r, El(l), c & 4 && u !== null && u.memoizedState.isDehydrated)
          try {
            ou(n.containerInfo);
          } catch (B) {
            Tt(l, l.return, B);
          }
        Ad && (Ad = !1, em(l));
        break;
      case 4:
        c = Pl, Pl = Qd(
          l.stateNode.containerInfo
        ), Il(n, l), El(l), Pl = c;
        break;
      case 12:
        Il(n, l), El(l);
        break;
      case 13:
        Il(n, l), El(l), l.child.flags & 8192 && l.memoizedState !== null != (u !== null && u.memoizedState !== null) && (om = vl()), c & 4 && (c = l.updateQueue, c !== null && (l.updateQueue = null, zd(l, c)));
        break;
      case 22:
        r = l.memoizedState !== null;
        var S = u !== null && u.memoizedState !== null, C = In, Z = kt;
        if (In = C || r, kt = Z || S, Il(n, l), kt = Z, In = C, El(l), c & 8192)
          e: for (n = l.stateNode, n._visibility = r ? n._visibility & -2 : n._visibility | 1, r && (u === null || S || In || kt || jt(l)), u = null, n = l; ; ) {
            if (n.tag === 5 || n.tag === 26) {
              if (u === null) {
                S = u = n;
                try {
                  if (s = S.stateNode, r)
                    y = s.style, typeof y.setProperty == "function" ? y.setProperty("display", "none", "important") : y.display = "none";
                  else {
                    p = S.stateNode;
                    var W = S.memoizedProps.style, N = W != null && W.hasOwnProperty("display") ? W.display : null;
                    p.style.display = N == null || typeof N == "boolean" ? "" : ("" + N).trim();
                  }
                } catch (B) {
                  Tt(S, S.return, B);
                }
              }
            } else if (n.tag === 6) {
              if (u === null) {
                S = n;
                try {
                  S.stateNode.nodeValue = r ? "" : S.memoizedProps;
                } catch (B) {
                  Tt(S, S.return, B);
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
        c & 4 && (c = l.updateQueue, c !== null && (u = c.retryQueue, u !== null && (c.retryQueue = null, zd(l, u))));
        break;
      case 19:
        Il(n, l), El(l), c & 4 && (c = l.updateQueue, c !== null && (l.updateQueue = null, zd(l, c)));
        break;
      case 30:
        break;
      case 21:
        break;
      default:
        Il(n, l), El(l);
    }
  }
  function El(l) {
    var n = l.flags;
    if (n & 2) {
      try {
        for (var u, c = l.return; c !== null; ) {
          if (rv(c)) {
            u = c;
            break;
          }
          c = c.return;
        }
        if (u == null) throw Error(_(160));
        switch (u.tag) {
          case 27:
            var r = u.stateNode, s = Wa(l);
            Ed(l, s, r);
            break;
          case 5:
            var y = u.stateNode;
            u.flags & 32 && (io(y, ""), u.flags &= -33);
            var p = Wa(l);
            Ed(l, p, y);
            break;
          case 3:
          case 4:
            var S = u.stateNode.containerInfo, C = Wa(l);
            zc(
              l,
              C,
              S
            );
            break;
          default:
            throw Error(_(161));
        }
      } catch (Z) {
        Tt(l, l.return, Z);
      }
      l.flags &= -3;
    }
    n & 4096 && (l.flags &= -4097);
  }
  function em(l) {
    if (l.subtreeFlags & 1024)
      for (l = l.child; l !== null; ) {
        var n = l;
        em(n), n.tag === 5 && n.flags & 1024 && n.stateNode.reset(), l = l.sibling;
      }
  }
  function eu(l, n) {
    if (n.subtreeFlags & 8772)
      for (n = n.child; n !== null; )
        Fy(l, n.alternate, n), n = n.sibling;
  }
  function jt(l) {
    for (l = l.child; l !== null; ) {
      var n = l;
      switch (n.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          Oi(4, n, n.return), jt(n);
          break;
        case 1:
          Tn(n, n.return);
          var u = n.stateNode;
          typeof u.componentWillUnmount == "function" && ky(
            n,
            n.return,
            u
          ), jt(n);
          break;
        case 27:
          ga(n.stateNode);
        case 26:
        case 5:
          Tn(n, n.return), jt(n);
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
  function Di(l, n, u) {
    for (u = u && (n.subtreeFlags & 8772) !== 0, n = n.child; n !== null; ) {
      var c = n.alternate, r = l, s = n, y = s.flags;
      switch (s.tag) {
        case 0:
        case 11:
        case 15:
          Di(
            r,
            s,
            u
          ), vr(4, s);
          break;
        case 1:
          if (Di(
            r,
            s,
            u
          ), c = s, r = c.stateNode, typeof r.componentDidMount == "function")
            try {
              r.componentDidMount();
            } catch (C) {
              Tt(c, c.return, C);
            }
          if (c = s, r = c.updateQueue, r !== null) {
            var p = c.stateNode;
            try {
              var S = r.shared.hiddenCallbacks;
              if (S !== null)
                for (r.shared.hiddenCallbacks = null, r = 0; r < S.length; r++)
                  Xs(S[r], p);
            } catch (C) {
              Tt(c, c.return, C);
            }
          }
          u && y & 64 && Td(s), wo(s, s.return);
          break;
        case 27:
          Rd(s);
        case 26:
        case 5:
          Di(
            r,
            s,
            u
          ), u && c === null && y & 4 && qo(s), wo(s, s.return);
          break;
        case 12:
          Di(
            r,
            s,
            u
          );
          break;
        case 13:
          Di(
            r,
            s,
            u
          ), u && y & 4 && Dd(r, s);
          break;
        case 22:
          s.memoizedState === null && Di(
            r,
            s,
            u
          ), wo(s, s.return);
          break;
        case 30:
          break;
        default:
          Di(
            r,
            s,
            u
          );
      }
      n = n.sibling;
    }
  }
  function Fa(l, n) {
    var u = null;
    l !== null && l.memoizedState !== null && l.memoizedState.cachePool !== null && (u = l.memoizedState.cachePool.pool), l = null, n.memoizedState !== null && n.memoizedState.cachePool !== null && (l = n.memoizedState.cachePool.pool), l !== u && (l != null && l.refCount++, u != null && Ln(u));
  }
  function Md(l, n) {
    l = null, n.alternate !== null && (l = n.alternate.memoizedState.cache), n = n.memoizedState.cache, n !== l && (n.refCount++, l != null && Ln(l));
  }
  function Hl(l, n, u, c) {
    if (n.subtreeFlags & 10256)
      for (n = n.child; n !== null; )
        tm(
          l,
          n,
          u,
          c
        ), n = n.sibling;
  }
  function tm(l, n, u, c) {
    var r = n.flags;
    switch (n.tag) {
      case 0:
      case 11:
      case 15:
        Hl(
          l,
          n,
          u,
          c
        ), r & 2048 && vr(9, n);
        break;
      case 1:
        Hl(
          l,
          n,
          u,
          c
        );
        break;
      case 3:
        Hl(
          l,
          n,
          u,
          c
        ), r & 2048 && (l = null, n.alternate !== null && (l = n.alternate.memoizedState.cache), n = n.memoizedState.cache, n !== l && (n.refCount++, l != null && Ln(l)));
        break;
      case 12:
        if (r & 2048) {
          Hl(
            l,
            n,
            u,
            c
          ), l = n.stateNode;
          try {
            var s = n.memoizedProps, y = s.id, p = s.onPostCommit;
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
          Hl(
            l,
            n,
            u,
            c
          );
        break;
      case 13:
        Hl(
          l,
          n,
          u,
          c
        );
        break;
      case 23:
        break;
      case 22:
        s = n.stateNode, y = n.alternate, n.memoizedState !== null ? s._visibility & 2 ? Hl(
          l,
          n,
          u,
          c
        ) : ht(l, n) : s._visibility & 2 ? Hl(
          l,
          n,
          u,
          c
        ) : (s._visibility |= 2, Bu(
          l,
          n,
          u,
          c,
          (n.subtreeFlags & 10256) !== 0
        )), r & 2048 && Fa(y, n);
        break;
      case 24:
        Hl(
          l,
          n,
          u,
          c
        ), r & 2048 && Md(n.alternate, n);
        break;
      default:
        Hl(
          l,
          n,
          u,
          c
        );
    }
  }
  function Bu(l, n, u, c, r) {
    for (r = r && (n.subtreeFlags & 10256) !== 0, n = n.child; n !== null; ) {
      var s = l, y = n, p = u, S = c, C = y.flags;
      switch (y.tag) {
        case 0:
        case 11:
        case 15:
          Bu(
            s,
            y,
            p,
            S,
            r
          ), vr(8, y);
          break;
        case 23:
          break;
        case 22:
          var Z = y.stateNode;
          y.memoizedState !== null ? Z._visibility & 2 ? Bu(
            s,
            y,
            p,
            S,
            r
          ) : ht(
            s,
            y
          ) : (Z._visibility |= 2, Bu(
            s,
            y,
            p,
            S,
            r
          )), r && C & 2048 && Fa(
            y.alternate,
            y
          );
          break;
        case 24:
          Bu(
            s,
            y,
            p,
            S,
            r
          ), r && C & 2048 && Md(y.alternate, y);
          break;
        default:
          Bu(
            s,
            y,
            p,
            S,
            r
          );
      }
      n = n.sibling;
    }
  }
  function ht(l, n) {
    if (n.subtreeFlags & 10256)
      for (n = n.child; n !== null; ) {
        var u = l, c = n, r = c.flags;
        switch (c.tag) {
          case 22:
            ht(u, c), r & 2048 && Fa(
              c.alternate,
              c
            );
            break;
          case 24:
            ht(u, c), r & 2048 && Md(c.alternate, c);
            break;
          default:
            ht(u, c);
        }
        n = n.sibling;
      }
  }
  var Mc = 8192;
  function $t(l) {
    if (l.subtreeFlags & Mc)
      for (l = l.child; l !== null; )
        sv(l), l = l.sibling;
  }
  function sv(l) {
    switch (l.tag) {
      case 26:
        $t(l), l.flags & Mc && l.memoizedState !== null && Uv(
          Pl,
          l.memoizedState,
          l.memoizedProps
        );
        break;
      case 5:
        $t(l);
        break;
      case 3:
      case 4:
        var n = Pl;
        Pl = Qd(l.stateNode.containerInfo), $t(l), Pl = n;
        break;
      case 22:
        l.memoizedState === null && (n = l.alternate, n !== null && n.memoizedState !== null ? (n = Mc, Mc = 16777216, $t(l), Mc = n) : $t(l));
        break;
      default:
        $t(l);
    }
  }
  function lm(l) {
    var n = l.alternate;
    if (n !== null && (l = n.child, l !== null)) {
      n.child = null;
      do
        n = l.sibling, l.sibling = null, l = n;
      while (l !== null);
    }
  }
  function _c(l) {
    var n = l.deletions;
    if ((l.flags & 16) !== 0) {
      if (n !== null)
        for (var u = 0; u < n.length; u++) {
          var c = n[u];
          hl = c, nm(
            c,
            l
          );
        }
      lm(l);
    }
    if (l.subtreeFlags & 10256)
      for (l = l.child; l !== null; )
        am(l), l = l.sibling;
  }
  function am(l) {
    switch (l.tag) {
      case 0:
      case 11:
      case 15:
        _c(l), l.flags & 2048 && Oi(9, l, l.return);
        break;
      case 3:
        _c(l);
        break;
      case 12:
        _c(l);
        break;
      case 22:
        var n = l.stateNode;
        l.memoizedState !== null && n._visibility & 2 && (l.return === null || l.return.tag !== 13) ? (n._visibility &= -3, ea(l)) : _c(l);
        break;
      default:
        _c(l);
    }
  }
  function ea(l) {
    var n = l.deletions;
    if ((l.flags & 16) !== 0) {
      if (n !== null)
        for (var u = 0; u < n.length; u++) {
          var c = n[u];
          hl = c, nm(
            c,
            l
          );
        }
      lm(l);
    }
    for (l = l.child; l !== null; ) {
      switch (n = l, n.tag) {
        case 0:
        case 11:
        case 15:
          Oi(8, n, n.return), ea(n);
          break;
        case 22:
          u = n.stateNode, u._visibility & 2 && (u._visibility &= -3, ea(n));
          break;
        default:
          ea(n);
      }
      l = l.sibling;
    }
  }
  function nm(l, n) {
    for (; hl !== null; ) {
      var u = hl;
      switch (u.tag) {
        case 0:
        case 11:
        case 15:
          Oi(8, u, n);
          break;
        case 23:
        case 22:
          if (u.memoizedState !== null && u.memoizedState.cachePool !== null) {
            var c = u.memoizedState.cachePool.pool;
            c != null && c.refCount++;
          }
          break;
        case 24:
          Ln(u.memoizedState.cache);
      }
      if (c = u.child, c !== null) c.return = u, hl = c;
      else
        e: for (u = l; hl !== null; ) {
          c = hl;
          var r = c.sibling, s = c.return;
          if (Iy(c), c === u) {
            hl = null;
            break e;
          }
          if (r !== null) {
            r.return = s, hl = r;
            break e;
          }
          hl = s;
        }
    }
  }
  var um = {
    getCacheForType: function(l) {
      var n = bl(rl), u = n.data.get(l);
      return u === void 0 && (u = l(), n.data.set(l, u)), u;
    }
  }, dv = typeof WeakMap == "function" ? WeakMap : Map, gt = 0, Ut = null, tt = null, lt = 0, bt = 0, ma = null, tu = !1, Bo = !1, im = !1, Yu = 0, Wt = 0, ju = 0, Uc = 0, lu = 0, Ia = 0, Yo = 0, jo = null, pa = null, cm = !1, om = 0, _d = 1 / 0, Go = null, zi = null, Nl = 0, au = null, Lo = null, wl = 0, Ud = 0, Cd = null, fm = null, Vo = 0, rm = null;
  function Ua() {
    if ((gt & 2) !== 0 && lt !== 0)
      return lt & -lt;
    if (R.T !== null) {
      var l = Ja;
      return l !== 0 ? l : Nc();
    }
    return fs();
  }
  function sm() {
    Ia === 0 && (Ia = (lt & 536870912) === 0 || rt ? te() : 536870912);
    var l = _a.current;
    return l !== null && (l.flags |= 32), Ia;
  }
  function Ca(l, n, u) {
    (l === Ut && (bt === 2 || bt === 9) || l.cancelPendingCommit !== null) && (nu(l, 0), Gu(
      l,
      lt,
      Ia,
      !1
    )), we(l, u), ((gt & 2) === 0 || l !== Ut) && (l === Ut && ((gt & 2) === 0 && (Uc |= u), Wt === 4 && Gu(
      l,
      lt,
      Ia,
      !1
    )), va(l));
  }
  function Xo(l, n, u) {
    if ((gt & 6) !== 0) throw Error(_(327));
    var c = !u && (n & 124) === 0 && (n & l.expiredLanes) === 0 || m(l, n), r = c ? hm(l, n) : xd(l, n, !0), s = c;
    do {
      if (r === 0) {
        Bo && !c && Gu(l, n, 0, !1);
        break;
      } else {
        if (u = l.current.alternate, s && !hv(u)) {
          r = xd(l, n, !1), s = !1;
          continue;
        }
        if (r === 2) {
          if (s = n, l.errorRecoveryDisabledLanes & s)
            var y = 0;
          else
            y = l.pendingLanes & -536870913, y = y !== 0 ? y : y & 536870912 ? 536870912 : 0;
          if (y !== 0) {
            n = y;
            e: {
              var p = l;
              r = jo;
              var S = p.current.memoizedState.isDehydrated;
              if (S && (nu(p, y).flags |= 256), y = xd(
                p,
                y,
                !1
              ), y !== 2) {
                if (im && !S) {
                  p.errorRecoveryDisabledLanes |= s, Uc |= s, r = 4;
                  break e;
                }
                s = pa, pa = r, s !== null && (pa === null ? pa = s : pa.push.apply(
                  pa,
                  s
                ));
              }
              r = y;
            }
            if (s = !1, r !== 2) continue;
          }
        }
        if (r === 1) {
          nu(l, 0), Gu(l, n, 0, !0);
          break;
        }
        e: {
          switch (c = l, s = r, s) {
            case 0:
            case 1:
              throw Error(_(345));
            case 4:
              if ((n & 4194048) !== n) break;
            case 6:
              Gu(
                c,
                n,
                Ia,
                !tu
              );
              break e;
            case 2:
              pa = null;
              break;
            case 3:
            case 5:
              break;
            default:
              throw Error(_(329));
          }
          if ((n & 62914560) === n && (r = om + 300 - vl(), 10 < r)) {
            if (Gu(
              c,
              n,
              Ia,
              !tu
            ), cn(c, 0, !0) !== 0) break e;
            c.timeoutHandle = Vd(
              br.bind(
                null,
                c,
                u,
                pa,
                Go,
                cm,
                n,
                Ia,
                Uc,
                Yo,
                tu,
                s,
                2,
                -0,
                0
              ),
              r
            );
            break e;
          }
          br(
            c,
            u,
            pa,
            Go,
            cm,
            n,
            Ia,
            Uc,
            Yo,
            tu,
            s,
            0,
            -0,
            0
          );
        }
      }
      break;
    } while (!0);
    va(l);
  }
  function br(l, n, u, c, r, s, y, p, S, C, Z, W, N, B) {
    if (l.timeoutHandle = -1, W = n.subtreeFlags, (W & 8192 || (W & 16785408) === 16785408) && (tf = { stylesheets: null, count: 0, unsuspend: _v }, sv(n), W = _m(), W !== null)) {
      l.cancelPendingCommit = W(
        pv.bind(
          null,
          l,
          n,
          s,
          u,
          c,
          r,
          y,
          p,
          S,
          Z,
          1,
          N,
          B
        )
      ), Gu(l, s, y, !C);
      return;
    }
    pv(
      l,
      n,
      s,
      u,
      c,
      r,
      y,
      p,
      S
    );
  }
  function hv(l) {
    for (var n = l; ; ) {
      var u = n.tag;
      if ((u === 0 || u === 11 || u === 15) && n.flags & 16384 && (u = n.updateQueue, u !== null && (u = u.stores, u !== null)))
        for (var c = 0; c < u.length; c++) {
          var r = u[c], s = r.getSnapshot;
          r = r.value;
          try {
            if (!Cl(s(), r)) return !1;
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
  function Gu(l, n, u, c) {
    n &= ~lu, n &= ~Uc, l.suspendedLanes |= n, l.pingedLanes &= ~n, c && (l.warmLanes |= n), c = l.expirationTimes;
    for (var r = n; 0 < r; ) {
      var s = 31 - zl(r), y = 1 << s;
      c[s] = -1, r &= ~y;
    }
    u !== 0 && ut(l, u, n);
  }
  function Cc() {
    return (gt & 6) === 0 ? (Rr(0), !1) : !0;
  }
  function Mi() {
    if (tt !== null) {
      if (bt === 0)
        var l = tt.return;
      else
        l = tt, vn = Uu = null, Js(l), Tc = null, Uo = 0, l = tt;
      for (; l !== null; )
        Jy(l.alternate, l), l = l.return;
      tt = null;
    }
  }
  function nu(l, n) {
    var u = l.timeoutHandle;
    u !== -1 && (l.timeoutHandle = -1, Bg(u)), u = l.cancelPendingCommit, u !== null && (l.cancelPendingCommit = null, u()), Mi(), Ut = l, tt = u = mn(l.current, null), lt = n, bt = 0, ma = null, tu = !1, Bo = m(l, n), im = !1, Yo = Ia = lu = Uc = ju = Wt = 0, pa = jo = null, cm = !1, (n & 8) !== 0 && (n |= n & 32);
    var c = l.entangledLanes;
    if (c !== 0)
      for (l = l.entanglements, c &= n; 0 < c; ) {
        var r = 31 - zl(c), s = 1 << r;
        n |= l[r], c &= ~s;
      }
    return Yu = n, yn(), u;
  }
  function dm(l, n) {
    Le = null, R.H = od, n === gi || n === $f ? (n = py(), bt = 3) : n === Ys ? (n = py(), bt = 4) : bt = n === Jt ? 8 : n !== null && typeof n == "object" && typeof n.then == "function" ? 6 : 1, ma = n, tt === null && (Wt = 1, hr(
      l,
      Da(n, l.current)
    ));
  }
  function yv() {
    var l = R.H;
    return R.H = od, l === null ? od : l;
  }
  function xc() {
    var l = R.A;
    return R.A = um, l;
  }
  function Hc() {
    Wt = 4, tu || (lt & 4194048) !== lt && _a.current !== null || (Bo = !0), (ju & 134217727) === 0 && (Uc & 134217727) === 0 || Ut === null || Gu(
      Ut,
      lt,
      Ia,
      !1
    );
  }
  function xd(l, n, u) {
    var c = gt;
    gt |= 2;
    var r = yv(), s = xc();
    (Ut !== l || lt !== n) && (Go = null, nu(l, n)), n = !1;
    var y = Wt;
    e: do
      try {
        if (bt !== 0 && tt !== null) {
          var p = tt, S = ma;
          switch (bt) {
            case 8:
              Mi(), y = 6;
              break e;
            case 3:
            case 2:
            case 9:
            case 6:
              _a.current === null && (n = !0);
              var C = bt;
              if (bt = 0, ma = null, Qo(l, p, S, C), u && Bo) {
                y = 0;
                break e;
              }
              break;
            default:
              C = bt, bt = 0, ma = null, Qo(l, p, S, C);
          }
        }
        Hd(), y = Wt;
        break;
      } catch (Z) {
        dm(l, Z);
      }
    while (!0);
    return n && l.shellSuspendCounter++, vn = Uu = null, gt = c, R.H = r, R.A = s, tt === null && (Ut = null, lt = 0, yn()), y;
  }
  function Hd() {
    for (; tt !== null; ) mm(tt);
  }
  function hm(l, n) {
    var u = gt;
    gt |= 2;
    var c = yv(), r = xc();
    Ut !== l || lt !== n ? (Go = null, _d = vl() + 500, nu(l, n)) : Bo = m(
      l,
      n
    );
    e: do
      try {
        if (bt !== 0 && tt !== null) {
          n = tt;
          var s = ma;
          t: switch (bt) {
            case 1:
              bt = 0, ma = null, Qo(l, n, s, 1);
              break;
            case 2:
            case 9:
              if (Gs(s)) {
                bt = 0, ma = null, pm(n);
                break;
              }
              n = function() {
                bt !== 2 && bt !== 9 || Ut !== l || (bt = 7), va(l);
              }, s.then(n, n);
              break e;
            case 3:
              bt = 7;
              break e;
            case 4:
              bt = 5;
              break e;
            case 7:
              Gs(s) ? (bt = 0, ma = null, pm(n)) : (bt = 0, ma = null, Qo(l, n, s, 7));
              break;
            case 5:
              var y = null;
              switch (tt.tag) {
                case 26:
                  y = tt.memoizedState;
                case 5:
                case 27:
                  var p = tt;
                  if (!y || Mm(y)) {
                    bt = 0, ma = null;
                    var S = p.sibling;
                    if (S !== null) tt = S;
                    else {
                      var C = p.return;
                      C !== null ? (tt = C, Sr(C)) : tt = null;
                    }
                    break t;
                  }
              }
              bt = 0, ma = null, Qo(l, n, s, 5);
              break;
            case 6:
              bt = 0, ma = null, Qo(l, n, s, 6);
              break;
            case 8:
              Mi(), Wt = 6;
              break e;
            default:
              throw Error(_(462));
          }
        }
        ym();
        break;
      } catch (Z) {
        dm(l, Z);
      }
    while (!0);
    return vn = Uu = null, R.H = c, R.A = r, gt = u, tt !== null ? 0 : (Ut = null, lt = 0, yn(), Wt);
  }
  function ym() {
    for (; tt !== null && !Sf(); )
      mm(tt);
  }
  function mm(l) {
    var n = fv(l.alternate, l, Yu);
    l.memoizedProps = l.pendingProps, n === null ? Sr(l) : tt = n;
  }
  function pm(l) {
    var n = l, u = n.alternate;
    switch (n.tag) {
      case 15:
      case 0:
        n = Vy(
          u,
          n,
          n.pendingProps,
          n.type,
          void 0,
          lt
        );
        break;
      case 11:
        n = Vy(
          u,
          n,
          n.pendingProps,
          n.type.render,
          n.ref,
          lt
        );
        break;
      case 5:
        Js(n);
      default:
        Jy(u, n), n = tt = $e(n, Yu), n = fv(u, n, Yu);
    }
    l.memoizedProps = l.pendingProps, n === null ? Sr(l) : tt = n;
  }
  function Qo(l, n, u, c) {
    vn = Uu = null, Js(n), Tc = null, Uo = 0;
    var r = n.return;
    try {
      if (iv(
        l,
        r,
        n,
        u,
        lt
      )) {
        Wt = 1, hr(
          l,
          Da(u, l.current)
        ), tt = null;
        return;
      }
    } catch (s) {
      if (r !== null) throw tt = r, s;
      Wt = 1, hr(
        l,
        Da(u, l.current)
      ), tt = null;
      return;
    }
    n.flags & 32768 ? (rt || c === 1 ? l = !0 : Bo || (lt & 536870912) !== 0 ? l = !1 : (tu = l = !0, (c === 2 || c === 9 || c === 3 || c === 6) && (c = _a.current, c !== null && c.tag === 13 && (c.flags |= 16384))), mv(n, l)) : Sr(n);
  }
  function Sr(l) {
    var n = l;
    do {
      if ((n.flags & 32768) !== 0) {
        mv(
          n,
          tu
        );
        return;
      }
      l = n.return;
      var u = Ky(
        n.alternate,
        n,
        Yu
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
    Wt === 0 && (Wt = 5);
  }
  function mv(l, n) {
    do {
      var u = Cg(l.alternate, l);
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
    Wt = 6, tt = null;
  }
  function pv(l, n, u, c, r, s, y, p, S) {
    l.cancelPendingCommit = null;
    do
      wd();
    while (Nl !== 0);
    if ((gt & 6) !== 0) throw Error(_(327));
    if (n !== null) {
      if (n === l.current) throw Error(_(177));
      if (s = n.lanes | n.childLanes, s |= Bn, Ge(
        l,
        u,
        s,
        y,
        p,
        S
      ), l === Ut && (tt = Ut = null, lt = 0), Lo = n, au = l, wl = u, Ud = s, Cd = r, fm = c, (n.subtreeFlags & 10256) !== 0 || (n.flags & 10256) !== 0 ? (l.callbackNode = null, l.callbackPriority = 0, Ng(Un, function() {
        return vm(), null;
      })) : (l.callbackNode = null, l.callbackPriority = 0), c = (n.flags & 13878) !== 0, (n.subtreeFlags & 13878) !== 0 || c) {
        c = R.T, R.T = null, r = X.p, X.p = 2, y = gt, gt |= 4;
        try {
          Wy(l, n, u);
        } finally {
          gt = y, X.p = r, R.T = c;
        }
      }
      Nl = 1, vv(), Tr(), Nd();
    }
  }
  function vv() {
    if (Nl === 1) {
      Nl = 0;
      var l = au, n = Lo, u = (n.flags & 13878) !== 0;
      if ((n.subtreeFlags & 13878) !== 0 || u) {
        u = R.T, R.T = null;
        var c = X.p;
        X.p = 2;
        var r = gt;
        gt |= 4;
        try {
          gr(n, l);
          var s = Cr, y = uy(l.containerInfo), p = s.focusedElem, S = s.selectionRange;
          if (y !== p && p && p.ownerDocument && Yf(
            p.ownerDocument.documentElement,
            p
          )) {
            if (S !== null && jf(p)) {
              var C = S.start, Z = S.end;
              if (Z === void 0 && (Z = C), "selectionStart" in p)
                p.selectionStart = C, p.selectionEnd = Math.min(
                  Z,
                  p.value.length
                );
              else {
                var W = p.ownerDocument || document, N = W && W.defaultView || window;
                if (N.getSelection) {
                  var B = N.getSelection(), Te = p.textContent.length, Ee = Math.min(S.start, Te), yt = S.end === void 0 ? Ee : Math.min(S.end, Te);
                  !B.extend && Ee > yt && (y = yt, yt = Ee, Ee = y);
                  var M = Nt(
                    p,
                    Ee
                  ), O = Nt(
                    p,
                    yt
                  );
                  if (M && O && (B.rangeCount !== 1 || B.anchorNode !== M.node || B.anchorOffset !== M.offset || B.focusNode !== O.node || B.focusOffset !== O.offset)) {
                    var U = W.createRange();
                    U.setStart(M.node, M.offset), B.removeAllRanges(), Ee > yt ? (B.addRange(U), B.extend(O.node, O.offset)) : (U.setEnd(O.node, O.offset), B.addRange(U));
                  }
                }
              }
            }
            for (W = [], B = p; B = B.parentNode; )
              B.nodeType === 1 && W.push({
                element: B,
                left: B.scrollLeft,
                top: B.scrollTop
              });
            for (typeof p.focus == "function" && p.focus(), p = 0; p < W.length; p++) {
              var $ = W[p];
              $.element.scrollLeft = $.left, $.element.scrollTop = $.top;
            }
          }
          qr = !!Ur, Cr = Ur = null;
        } finally {
          gt = r, X.p = c, R.T = u;
        }
      }
      l.current = n, Nl = 2;
    }
  }
  function Tr() {
    if (Nl === 2) {
      Nl = 0;
      var l = au, n = Lo, u = (n.flags & 8772) !== 0;
      if ((n.subtreeFlags & 8772) !== 0 || u) {
        u = R.T, R.T = null;
        var c = X.p;
        X.p = 2;
        var r = gt;
        gt |= 4;
        try {
          Fy(l, n.alternate, n);
        } finally {
          gt = r, X.p = c, R.T = u;
        }
      }
      Nl = 3;
    }
  }
  function Nd() {
    if (Nl === 4 || Nl === 3) {
      Nl = 0, ll();
      var l = au, n = Lo, u = wl, c = fm;
      (n.subtreeFlags & 10256) !== 0 || (n.flags & 10256) !== 0 ? Nl = 5 : (Nl = 0, Lo = au = null, gv(l, l.pendingLanes));
      var r = l.pendingLanes;
      if (r === 0 && (zi = null), on(u), n = n.stateNode, Dl && typeof Dl.onCommitFiberRoot == "function")
        try {
          Dl.onCommitFiberRoot(
            ei,
            n,
            void 0,
            (n.current.flags & 128) === 128
          );
        } catch {
        }
      if (c !== null) {
        n = R.T, r = X.p, X.p = 2, R.T = null;
        try {
          for (var s = l.onRecoverableError, y = 0; y < c.length; y++) {
            var p = c[y];
            s(p.value, {
              componentStack: p.stack
            });
          }
        } finally {
          R.T = n, X.p = r;
        }
      }
      (wl & 3) !== 0 && wd(), va(l), r = l.pendingLanes, (u & 4194090) !== 0 && (r & 42) !== 0 ? l === rm ? Vo++ : (Vo = 0, rm = l) : Vo = 0, Rr(0);
    }
  }
  function gv(l, n) {
    (l.pooledCacheLanes &= n) === 0 && (n = l.pooledCache, n != null && (l.pooledCache = null, Ln(n)));
  }
  function wd(l) {
    return vv(), Tr(), Nd(), vm();
  }
  function vm() {
    if (Nl !== 5) return !1;
    var l = au, n = Ud;
    Ud = 0;
    var u = on(wl), c = R.T, r = X.p;
    try {
      X.p = 32 > u ? 32 : u, R.T = null, u = Cd, Cd = null;
      var s = au, y = wl;
      if (Nl = 0, Lo = au = null, wl = 0, (gt & 6) !== 0) throw Error(_(331));
      var p = gt;
      if (gt |= 4, am(s.current), tm(
        s,
        s.current,
        y,
        u
      ), gt = p, Rr(0, !1), Dl && typeof Dl.onPostCommitFiberRoot == "function")
        try {
          Dl.onPostCommitFiberRoot(ei, s);
        } catch {
        }
      return !0;
    } finally {
      X.p = r, R.T = c, gv(l, n);
    }
  }
  function gm(l, n, u) {
    n = Da(u, n), n = Gy(l.stateNode, n, 2), l = Qn(l, n, 2), l !== null && (we(l, 2), va(l));
  }
  function Tt(l, n, u) {
    if (l.tag === 3)
      gm(l, l, u);
    else
      for (; n !== null; ) {
        if (n.tag === 3) {
          gm(
            n,
            l,
            u
          );
          break;
        } else if (n.tag === 1) {
          var c = n.stateNode;
          if (typeof n.type.getDerivedStateFromError == "function" || typeof c.componentDidCatch == "function" && (zi === null || !zi.has(c))) {
            l = Da(u, l), u = Ly(2), c = Qn(n, u, 2), c !== null && (ya(
              u,
              c,
              n,
              l
            ), we(c, 2), va(c));
            break;
          }
        }
        n = n.return;
      }
  }
  function qd(l, n, u) {
    var c = l.pingCache;
    if (c === null) {
      c = l.pingCache = new dv();
      var r = /* @__PURE__ */ new Set();
      c.set(n, r);
    } else
      r = c.get(n), r === void 0 && (r = /* @__PURE__ */ new Set(), c.set(n, r));
    r.has(u) || (im = !0, r.add(u), l = bm.bind(null, l, n, u), n.then(l, l));
  }
  function bm(l, n, u) {
    var c = l.pingCache;
    c !== null && c.delete(n), l.pingedLanes |= l.suspendedLanes & u, l.warmLanes &= ~u, Ut === l && (lt & u) === u && (Wt === 4 || Wt === 3 && (lt & 62914560) === lt && 300 > vl() - om ? (gt & 2) === 0 && nu(l, 0) : lu |= u, Yo === lt && (Yo = 0)), va(l);
  }
  function Sm(l, n) {
    n === 0 && (n = ne()), l = Yn(l, n), l !== null && (we(l, n), va(l));
  }
  function xg(l) {
    var n = l.memoizedState, u = 0;
    n !== null && (u = n.retryLane), Sm(l, u);
  }
  function Hg(l, n) {
    var u = 0;
    switch (l.tag) {
      case 13:
        var c = l.stateNode, r = l.memoizedState;
        r !== null && (u = r.retryLane);
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
    c !== null && c.delete(n), Sm(l, u);
  }
  function Ng(l, n) {
    return _n(l, n);
  }
  var Bd = null, _i = null, Er = !1, Zo = !1, Yd = !1, Ui = 0;
  function va(l) {
    l !== _i && l.next === null && (_i === null ? Bd = _i = l : _i = _i.next = l), Zo = !0, Er || (Er = !0, Tv());
  }
  function Rr(l, n) {
    if (!Yd && Zo) {
      Yd = !0;
      do
        for (var u = !1, c = Bd; c !== null; ) {
          if (l !== 0) {
            var r = c.pendingLanes;
            if (r === 0) var s = 0;
            else {
              var y = c.suspendedLanes, p = c.pingedLanes;
              s = (1 << 31 - zl(42 | l) + 1) - 1, s &= r & ~(y & ~p), s = s & 201326741 ? s & 201326741 | 1 : s ? s | 2 : 0;
            }
            s !== 0 && (u = !0, Or(c, s));
          } else
            s = lt, s = cn(
              c,
              c === Ut ? s : 0,
              c.cancelPendingCommit !== null || c.timeoutHandle !== -1
            ), (s & 3) === 0 || m(c, s) || (u = !0, Or(c, s));
          c = c.next;
        }
      while (u);
      Yd = !1;
    }
  }
  function bv() {
    Ar();
  }
  function Ar() {
    Zo = Er = !1;
    var l = 0;
    Ui !== 0 && (Xu() && (l = Ui), Ui = 0);
    for (var n = vl(), u = null, c = Bd; c !== null; ) {
      var r = c.next, s = Tm(c, n);
      s === 0 ? (c.next = null, u === null ? Bd = r : u.next = r, r === null && (_i = u)) : (u = c, (l !== 0 || (s & 3) !== 0) && (Zo = !0)), c = r;
    }
    Rr(l);
  }
  function Tm(l, n) {
    for (var u = l.suspendedLanes, c = l.pingedLanes, r = l.expirationTimes, s = l.pendingLanes & -62914561; 0 < s; ) {
      var y = 31 - zl(s), p = 1 << y, S = r[y];
      S === -1 ? ((p & u) === 0 || (p & c) !== 0) && (r[y] = z(p, n)) : S <= n && (l.expiredLanes |= p), s &= ~p;
    }
    if (n = Ut, u = lt, u = cn(
      l,
      l === n ? u : 0,
      l.cancelPendingCommit !== null || l.timeoutHandle !== -1
    ), c = l.callbackNode, u === 0 || l === n && (bt === 2 || bt === 9) || l.cancelPendingCommit !== null)
      return c !== null && c !== null && eo(c), l.callbackNode = null, l.callbackPriority = 0;
    if ((u & 3) === 0 || m(l, u)) {
      if (n = u & -u, n === l.callbackPriority) return n;
      switch (c !== null && eo(c), on(u)) {
        case 2:
        case 8:
          u = Je;
          break;
        case 32:
          u = Un;
          break;
        case 268435456:
          u = Su;
          break;
        default:
          u = Un;
      }
      return c = Sv.bind(null, l), u = _n(u, c), l.callbackPriority = n, l.callbackNode = u, n;
    }
    return c !== null && c !== null && eo(c), l.callbackPriority = 2, l.callbackNode = null, 2;
  }
  function Sv(l, n) {
    if (Nl !== 0 && Nl !== 5)
      return l.callbackNode = null, l.callbackPriority = 0, null;
    var u = l.callbackNode;
    if (wd() && l.callbackNode !== u)
      return null;
    var c = lt;
    return c = cn(
      l,
      l === Ut ? c : 0,
      l.cancelPendingCommit !== null || l.timeoutHandle !== -1
    ), c === 0 ? null : (Xo(l, c, n), Tm(l, vl()), l.callbackNode != null && l.callbackNode === u ? Sv.bind(null, l) : null);
  }
  function Or(l, n) {
    if (wd()) return null;
    Xo(l, n, !0);
  }
  function Tv() {
    Yg(function() {
      (gt & 6) !== 0 ? _n(
        cs,
        bv
      ) : Ar();
    });
  }
  function Nc() {
    return Ui === 0 && (Ui = te()), Ui;
  }
  function jd(l) {
    return l == null || typeof l == "symbol" || typeof l == "boolean" ? null : typeof l == "function" ? l : Uf("" + l);
  }
  function Dr(l, n) {
    var u = n.ownerDocument.createElement("input");
    return u.name = n.name, u.value = n.value, l.id && u.setAttribute("form", l.id), n.parentNode.insertBefore(u, n), l = new FormData(l), u.parentNode.removeChild(u), l;
  }
  function Ev(l, n, u, c, r) {
    if (n === "submit" && u && u.stateNode === r) {
      var s = jd(
        (r[$l] || null).action
      ), y = c.submitter;
      y && (n = (n = y[$l] || null) ? jd(n.formAction) : y.getAttribute("formAction"), n !== null && (s = n, y = null));
      var p = new Ts(
        "action",
        "action",
        null,
        c,
        r
      );
      l.push({
        event: p,
        listeners: [
          {
            instance: null,
            listener: function() {
              if (c.defaultPrevented) {
                if (Ui !== 0) {
                  var S = y ? Dr(r, y) : new FormData(r);
                  cd(
                    u,
                    {
                      pending: !0,
                      data: S,
                      method: r.method,
                      action: s
                    },
                    null,
                    S
                  );
                }
              } else
                typeof s == "function" && (p.preventDefault(), S = y ? Dr(r, y) : new FormData(r), cd(
                  u,
                  {
                    pending: !0,
                    data: S,
                    method: r.method,
                    action: s
                  },
                  s,
                  S
                ));
            },
            currentTarget: r
          }
        ]
      });
    }
  }
  for (var Ft = 0; Ft < ho.length; Ft++) {
    var zr = ho[Ft], wg = zr.toLowerCase(), ke = zr[0].toUpperCase() + zr.slice(1);
    Va(
      wg,
      "on" + ke
    );
  }
  Va(Zp, "onAnimationEnd"), Va(iy, "onAnimationIteration"), Va(Kp, "onAnimationStart"), Va("dblclick", "onDoubleClick"), Va("focusin", "onFocus"), Va("focusout", "onBlur"), Va(cy, "onTransitionRun"), Va(Us, "onTransitionStart"), Va(Jp, "onTransitionCancel"), Va(oy, "onTransitionEnd"), li("onMouseEnter", ["mouseout", "mouseover"]), li("onMouseLeave", ["mouseout", "mouseover"]), li("onPointerEnter", ["pointerout", "pointerover"]), li("onPointerLeave", ["pointerout", "pointerover"]), ti(
    "onChange",
    "change click focusin focusout input keydown keyup selectionchange".split(" ")
  ), ti(
    "onSelect",
    "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(
      " "
    )
  ), ti("onBeforeInput", [
    "compositionend",
    "keypress",
    "textInput",
    "paste"
  ]), ti(
    "onCompositionEnd",
    "compositionend focusout keydown keypress keyup mousedown".split(" ")
  ), ti(
    "onCompositionStart",
    "compositionstart focusout keydown keypress keyup mousedown".split(" ")
  ), ti(
    "onCompositionUpdate",
    "compositionupdate focusout keydown keypress keyup mousedown".split(" ")
  );
  var Mr = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(
    " "
  ), Ci = new Set(
    "beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(Mr)
  );
  function wc(l, n) {
    n = (n & 4) !== 0;
    for (var u = 0; u < l.length; u++) {
      var c = l[u], r = c.event;
      c = c.listeners;
      e: {
        var s = void 0;
        if (n)
          for (var y = c.length - 1; 0 <= y; y--) {
            var p = c[y], S = p.instance, C = p.currentTarget;
            if (p = p.listener, S !== s && r.isPropagationStopped())
              break e;
            s = p, r.currentTarget = C;
            try {
              s(r);
            } catch (Z) {
              sr(Z);
            }
            r.currentTarget = null, s = S;
          }
        else
          for (y = 0; y < c.length; y++) {
            if (p = c[y], S = p.instance, C = p.currentTarget, p = p.listener, S !== s && r.isPropagationStopped())
              break e;
            s = p, r.currentTarget = C;
            try {
              s(r);
            } catch (Z) {
              sr(Z);
            }
            r.currentTarget = null, s = S;
          }
      }
    }
  }
  function Ve(l, n) {
    var u = n[rs];
    u === void 0 && (u = n[rs] = /* @__PURE__ */ new Set());
    var c = l + "__bubble";
    u.has(c) || (Gd(n, l, 2, !1), u.add(c));
  }
  function Ko(l, n, u) {
    var c = 0;
    n && (c |= 4), Gd(
      u,
      l,
      c,
      n
    );
  }
  var Jo = "_reactListening" + Math.random().toString(36).slice(2);
  function Em(l) {
    if (!l[Jo]) {
      l[Jo] = !0, Of.forEach(function(u) {
        u !== "selectionchange" && (Ci.has(u) || Ko(u, !1, l), Ko(u, !0, l));
      });
      var n = l.nodeType === 9 ? l : l.ownerDocument;
      n === null || n[Jo] || (n[Jo] = !0, Ko("selectionchange", !1, n));
    }
  }
  function Gd(l, n, u, c) {
    switch (Bm(n)) {
      case 2:
        var r = Cv;
        break;
      case 8:
        r = xv;
        break;
      default:
        r = wm;
    }
    u = r.bind(
      null,
      n,
      u,
      l
    ), r = void 0, !gs || n !== "touchstart" && n !== "touchmove" && n !== "wheel" || (r = !0), c ? r !== void 0 ? l.addEventListener(n, u, {
      capture: !0,
      passive: r
    }) : l.addEventListener(n, u, !0) : r !== void 0 ? l.addEventListener(n, u, {
      passive: r
    }) : l.addEventListener(n, u, !1);
  }
  function Pa(l, n, u, c, r) {
    var s = c;
    if ((n & 1) === 0 && (n & 2) === 0 && c !== null)
      e: for (; ; ) {
        if (c === null) return;
        var y = c.tag;
        if (y === 3 || y === 4) {
          var p = c.stateNode.containerInfo;
          if (p === r) break;
          if (y === 4)
            for (y = c.return; y !== null; ) {
              var S = y.tag;
              if ((S === 3 || S === 4) && y.stateNode.containerInfo === r)
                return;
              y = y.return;
            }
          for (; p !== null; ) {
            if (y = _l(p), y === null) return;
            if (S = y.tag, S === 5 || S === 6 || S === 26 || S === 27) {
              c = s = y;
              continue e;
            }
            p = p.parentNode;
          }
        }
        c = c.return;
      }
    fo(function() {
      var C = s, Z = vs(u), W = [];
      e: {
        var N = fy.get(l);
        if (N !== void 0) {
          var B = Ts, Te = l;
          switch (l) {
            case "keypress":
              if (Ul(u) === 0) break e;
            case "keydown":
            case "keyup":
              B = fn;
              break;
            case "focusin":
              Te = "focus", B = Qh;
              break;
            case "focusout":
              Te = "blur", B = Qh;
              break;
            case "beforeblur":
            case "afterblur":
              B = Qh;
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
              B = Xh;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              B = qp;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              B = Jh;
              break;
            case Zp:
            case iy:
            case Kp:
              B = zg;
              break;
            case oy:
              B = Lp;
              break;
            case "scroll":
            case "scrollend":
              B = Np;
              break;
            case "wheel":
              B = nc;
              break;
            case "copy":
            case "cut":
            case "paste":
              B = Hf;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              B = Nf;
              break;
            case "toggle":
            case "beforetoggle":
              B = Vp;
          }
          var Ee = (n & 4) !== 0, yt = !Ee && (l === "scroll" || l === "scrollend"), M = Ee ? N !== null ? N + "Capture" : null : N;
          Ee = [];
          for (var O = C, U; O !== null; ) {
            var $ = O;
            if (U = $.stateNode, $ = $.tag, $ !== 5 && $ !== 26 && $ !== 27 || U === null || M === null || ($ = ec(O, M), $ != null && Ee.push(
              Lu(O, $, U)
            )), yt) break;
            O = O.return;
          }
          0 < Ee.length && (N = new B(
            N,
            Te,
            null,
            u,
            Z
          ), W.push({ event: N, listeners: Ee }));
        }
      }
      if ((n & 7) === 0) {
        e: {
          if (N = l === "mouseover" || l === "pointerover", B = l === "mouseout" || l === "pointerout", N && u !== Pi && (Te = u.relatedTarget || u.fromElement) && (_l(Te) || Te[no]))
            break e;
          if ((B || N) && (N = Z.window === Z ? Z : (N = Z.ownerDocument) ? N.defaultView || N.parentWindow : window, B ? (Te = u.relatedTarget || u.toElement, B = C, Te = Te ? _l(Te) : null, Te !== null && (yt = Ae(Te), Ee = Te.tag, Te !== yt || Ee !== 5 && Ee !== 27 && Ee !== 6) && (Te = null)) : (B = null, Te = C), B !== Te)) {
            if (Ee = Xh, $ = "onMouseLeave", M = "onMouseEnter", O = "mouse", (l === "pointerout" || l === "pointerover") && (Ee = Nf, $ = "onPointerLeave", M = "onPointerEnter", O = "pointer"), yt = B == null ? N : Af(B), U = Te == null ? N : Af(Te), N = new Ee(
              $,
              O + "leave",
              B,
              u,
              Z
            ), N.target = yt, N.relatedTarget = U, $ = null, _l(Z) === C && (Ee = new Ee(
              M,
              O + "enter",
              Te,
              u,
              Z
            ), Ee.target = U, Ee.relatedTarget = yt, $ = Ee), yt = $, B && Te)
              t: {
                for (Ee = B, M = Te, O = 0, U = Ee; U; U = xi(U))
                  O++;
                for (U = 0, $ = M; $; $ = xi($))
                  U++;
                for (; 0 < O - U; )
                  Ee = xi(Ee), O--;
                for (; 0 < U - O; )
                  M = xi(M), U--;
                for (; O--; ) {
                  if (Ee === M || M !== null && Ee === M.alternate)
                    break t;
                  Ee = xi(Ee), M = xi(M);
                }
                Ee = null;
              }
            else Ee = null;
            B !== null && _r(
              W,
              N,
              B,
              Ee,
              !1
            ), Te !== null && yt !== null && _r(
              W,
              yt,
              Te,
              Ee,
              !0
            );
          }
        }
        e: {
          if (N = C ? Af(C) : window, B = N.nodeName && N.nodeName.toLowerCase(), B === "select" || B === "input" && N.type === "file")
            var se = Ph;
          else if (Ds(N))
            if (ey)
              se = ay;
            else {
              se = oi;
              var We = Ms;
            }
          else
            B = N.nodeName, !B || B.toLowerCase() !== "input" || N.type !== "checkbox" && N.type !== "radio" ? C && Ii(C.elementType) && (se = Ph) : se = Du;
          if (se && (se = se(l, C))) {
            zs(
              W,
              se,
              u,
              Z
            );
            break e;
          }
          We && We(l, N, C), l === "focusout" && C && N.type === "number" && C.memoizedProps.value != null && Mf(N, "number", N.value);
        }
        switch (We = C ? Af(C) : window, l) {
          case "focusin":
            (Ds(We) || We.contentEditable === "true") && (wn = We, dn = C, si = null);
            break;
          case "focusout":
            si = dn = wn = null;
            break;
          case "mousedown":
            fc = !0;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            fc = !1, _s(W, u, Z);
            break;
          case "selectionchange":
            if (oc) break;
          case "keydown":
          case "keyup":
            _s(W, u, Z);
        }
        var Se;
        if (wf)
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
          ci ? Bf(l, u) && (_e = "onCompositionEnd") : l === "keydown" && u.keyCode === 229 && (_e = "onCompositionStart");
        _e && (Hn && u.locale !== "ko" && (ci || _e !== "onCompositionStart" ? _e === "onCompositionEnd" && ci && (Se = Lh()) : (Au = Z, ro = "value" in Au ? Au.value : Au.textContent, ci = !0)), We = ko(C, _e), 0 < We.length && (_e = new Zh(
          _e,
          l,
          null,
          u,
          Z
        ), W.push({ event: _e, listeners: We }), Se ? _e.data = Se : (Se = ii(u), Se !== null && (_e.data = Se)))), (Se = $h ? Fh(l, u) : uc(l, u)) && (_e = ko(C, "onBeforeInput"), 0 < _e.length && (We = new Zh(
          "onBeforeInput",
          "beforeinput",
          null,
          u,
          Z
        ), W.push({
          event: We,
          listeners: _e
        }), We.data = Se)), Ev(
          W,
          l,
          C,
          u,
          Z
        );
      }
      wc(W, n);
    });
  }
  function Lu(l, n, u) {
    return {
      instance: l,
      listener: n,
      currentTarget: u
    };
  }
  function ko(l, n) {
    for (var u = n + "Capture", c = []; l !== null; ) {
      var r = l, s = r.stateNode;
      if (r = r.tag, r !== 5 && r !== 26 && r !== 27 || s === null || (r = ec(l, u), r != null && c.unshift(
        Lu(l, r, s)
      ), r = ec(l, n), r != null && c.push(
        Lu(l, r, s)
      )), l.tag === 3) return c;
      l = l.return;
    }
    return [];
  }
  function xi(l) {
    if (l === null) return null;
    do
      l = l.return;
    while (l && l.tag !== 5 && l.tag !== 27);
    return l || null;
  }
  function _r(l, n, u, c, r) {
    for (var s = n._reactName, y = []; u !== null && u !== c; ) {
      var p = u, S = p.alternate, C = p.stateNode;
      if (p = p.tag, S !== null && S === c) break;
      p !== 5 && p !== 26 && p !== 27 || C === null || (S = C, r ? (C = ec(u, s), C != null && y.unshift(
        Lu(u, C, S)
      )) : r || (C = ec(u, s), C != null && y.push(
        Lu(u, C, S)
      ))), u = u.return;
    }
    y.length !== 0 && l.push({ event: n, listeners: y });
  }
  var xa = /\r\n?/g, Rm = /\u0000|\uFFFD/g;
  function Rv(l) {
    return (typeof l == "string" ? l : "" + l).replace(xa, `
`).replace(Rm, "");
  }
  function Am(l, n) {
    return n = Rv(n), Rv(l) === n;
  }
  function Ld() {
  }
  function qe(l, n, u, c, r, s) {
    switch (u) {
      case "children":
        typeof c == "string" ? n === "body" || n === "textarea" && c === "" || io(l, c) : (typeof c == "number" || typeof c == "bigint") && n !== "body" && io(l, "" + c);
        break;
      case "className":
        Df(l, "class", c);
        break;
      case "tabIndex":
        Df(l, "tabindex", c);
        break;
      case "dir":
      case "role":
      case "viewBox":
      case "width":
      case "height":
        Df(l, u, c);
        break;
      case "style":
        _f(l, c, s);
        break;
      case "data":
        if (n !== "object") {
          Df(l, "data", c);
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
        c = Uf("" + c), l.setAttribute(u, c);
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
          typeof s == "function" && (u === "formAction" ? (n !== "input" && qe(l, n, "name", r.name, r, null), qe(
            l,
            n,
            "formEncType",
            r.formEncType,
            r,
            null
          ), qe(
            l,
            n,
            "formMethod",
            r.formMethod,
            r,
            null
          ), qe(
            l,
            n,
            "formTarget",
            r.formTarget,
            r,
            null
          )) : (qe(l, n, "encType", r.encType, r, null), qe(l, n, "method", r.method, r, null), qe(l, n, "target", r.target, r, null)));
        if (c == null || typeof c == "symbol" || typeof c == "boolean") {
          l.removeAttribute(u);
          break;
        }
        c = Uf("" + c), l.setAttribute(u, c);
        break;
      case "onClick":
        c != null && (l.onclick = Ld);
        break;
      case "onScroll":
        c != null && Ve("scroll", l);
        break;
      case "onScrollEnd":
        c != null && Ve("scrollend", l);
        break;
      case "dangerouslySetInnerHTML":
        if (c != null) {
          if (typeof c != "object" || !("__html" in c))
            throw Error(_(61));
          if (u = c.__html, u != null) {
            if (r.children != null) throw Error(_(60));
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
        u = Uf("" + c), l.setAttributeNS(
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
        Ve("beforetoggle", l), Ve("toggle", l), Eu(l, "popover", c);
        break;
      case "xlinkActuate":
        Cn(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:actuate",
          c
        );
        break;
      case "xlinkArcrole":
        Cn(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:arcrole",
          c
        );
        break;
      case "xlinkRole":
        Cn(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:role",
          c
        );
        break;
      case "xlinkShow":
        Cn(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:show",
          c
        );
        break;
      case "xlinkTitle":
        Cn(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:title",
          c
        );
        break;
      case "xlinkType":
        Cn(
          l,
          "http://www.w3.org/1999/xlink",
          "xlink:type",
          c
        );
        break;
      case "xmlBase":
        Cn(
          l,
          "http://www.w3.org/XML/1998/namespace",
          "xml:base",
          c
        );
        break;
      case "xmlLang":
        Cn(
          l,
          "http://www.w3.org/XML/1998/namespace",
          "xml:lang",
          c
        );
        break;
      case "xmlSpace":
        Cn(
          l,
          "http://www.w3.org/XML/1998/namespace",
          "xml:space",
          c
        );
        break;
      case "is":
        Eu(l, "is", c);
        break;
      case "innerText":
      case "textContent":
        break;
      default:
        (!(2 < u.length) || u[0] !== "o" && u[0] !== "O" || u[1] !== "n" && u[1] !== "N") && (u = Ag.get(u) || u, Eu(l, u, c));
    }
  }
  function Y(l, n, u, c, r, s) {
    switch (u) {
      case "style":
        _f(l, c, s);
        break;
      case "dangerouslySetInnerHTML":
        if (c != null) {
          if (typeof c != "object" || !("__html" in c))
            throw Error(_(61));
          if (u = c.__html, u != null) {
            if (r.children != null) throw Error(_(60));
            l.innerHTML = u;
          }
        }
        break;
      case "children":
        typeof c == "string" ? io(l, c) : (typeof c == "number" || typeof c == "bigint") && io(l, "" + c);
        break;
      case "onScroll":
        c != null && Ve("scroll", l);
        break;
      case "onScrollEnd":
        c != null && Ve("scrollend", l);
        break;
      case "onClick":
        c != null && (l.onclick = Ld);
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
            if (u[0] === "o" && u[1] === "n" && (r = u.endsWith("Capture"), n = u.slice(2, r ? u.length - 7 : void 0), s = l[$l] || null, s = s != null ? s[u] : null, typeof s == "function" && l.removeEventListener(n, s, r), typeof c == "function")) {
              typeof s != "function" && s !== null && (u in l ? l[u] = null : l.hasAttribute(u) && l.removeAttribute(u)), l.addEventListener(n, c, r);
              break e;
            }
            u in l ? l[u] = c : c === !0 ? l.setAttribute(u, "") : Eu(l, u, c);
          }
    }
  }
  function Ue(l, n, u) {
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
        Ve("error", l), Ve("load", l);
        var c = !1, r = !1, s;
        for (s in u)
          if (u.hasOwnProperty(s)) {
            var y = u[s];
            if (y != null)
              switch (s) {
                case "src":
                  c = !0;
                  break;
                case "srcSet":
                  r = !0;
                  break;
                case "children":
                case "dangerouslySetInnerHTML":
                  throw Error(_(137, n));
                default:
                  qe(l, n, s, y, u, null);
              }
          }
        r && qe(l, n, "srcSet", u.srcSet, u, null), c && qe(l, n, "src", u.src, u, null);
        return;
      case "input":
        Ve("invalid", l);
        var p = s = y = r = null, S = null, C = null;
        for (c in u)
          if (u.hasOwnProperty(c)) {
            var Z = u[c];
            if (Z != null)
              switch (c) {
                case "name":
                  r = Z;
                  break;
                case "type":
                  y = Z;
                  break;
                case "checked":
                  S = Z;
                  break;
                case "defaultChecked":
                  C = Z;
                  break;
                case "value":
                  s = Z;
                  break;
                case "defaultValue":
                  p = Z;
                  break;
                case "children":
                case "dangerouslySetInnerHTML":
                  if (Z != null)
                    throw Error(_(137, n));
                  break;
                default:
                  qe(l, n, c, Z, u, null);
              }
          }
        ms(
          l,
          s,
          p,
          S,
          C,
          y,
          r,
          !1
        ), ni(l);
        return;
      case "select":
        Ve("invalid", l), c = y = s = null;
        for (r in u)
          if (u.hasOwnProperty(r) && (p = u[r], p != null))
            switch (r) {
              case "value":
                s = p;
                break;
              case "defaultValue":
                y = p;
                break;
              case "multiple":
                c = p;
              default:
                qe(l, n, r, p, u, null);
            }
        n = s, u = y, l.multiple = !!c, n != null ? Fi(l, !!c, n, !1) : u != null && Fi(l, !!c, u, !0);
        return;
      case "textarea":
        Ve("invalid", l), s = r = c = null;
        for (y in u)
          if (u.hasOwnProperty(y) && (p = u[y], p != null))
            switch (y) {
              case "value":
                c = p;
                break;
              case "defaultValue":
                r = p;
                break;
              case "children":
                s = p;
                break;
              case "dangerouslySetInnerHTML":
                if (p != null) throw Error(_(91));
                break;
              default:
                qe(l, n, y, p, u, null);
            }
        jh(l, c, r, s), ni(l);
        return;
      case "option":
        for (S in u)
          if (u.hasOwnProperty(S) && (c = u[S], c != null))
            switch (S) {
              case "selected":
                l.selected = c && typeof c != "function" && typeof c != "symbol";
                break;
              default:
                qe(l, n, S, c, u, null);
            }
        return;
      case "dialog":
        Ve("beforetoggle", l), Ve("toggle", l), Ve("cancel", l), Ve("close", l);
        break;
      case "iframe":
      case "object":
        Ve("load", l);
        break;
      case "video":
      case "audio":
        for (c = 0; c < Mr.length; c++)
          Ve(Mr[c], l);
        break;
      case "image":
        Ve("error", l), Ve("load", l);
        break;
      case "details":
        Ve("toggle", l);
        break;
      case "embed":
      case "source":
      case "link":
        Ve("error", l), Ve("load", l);
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
        for (C in u)
          if (u.hasOwnProperty(C) && (c = u[C], c != null))
            switch (C) {
              case "children":
              case "dangerouslySetInnerHTML":
                throw Error(_(137, n));
              default:
                qe(l, n, C, c, u, null);
            }
        return;
      default:
        if (Ii(n)) {
          for (Z in u)
            u.hasOwnProperty(Z) && (c = u[Z], c !== void 0 && Y(
              l,
              n,
              Z,
              c,
              u,
              void 0
            ));
          return;
        }
    }
    for (p in u)
      u.hasOwnProperty(p) && (c = u[p], c != null && qe(l, n, p, c, u, null));
  }
  function qg(l, n, u, c) {
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
        var r = null, s = null, y = null, p = null, S = null, C = null, Z = null;
        for (B in u) {
          var W = u[B];
          if (u.hasOwnProperty(B) && W != null)
            switch (B) {
              case "checked":
                break;
              case "value":
                break;
              case "defaultValue":
                S = W;
              default:
                c.hasOwnProperty(B) || qe(l, n, B, null, c, W);
            }
        }
        for (var N in c) {
          var B = c[N];
          if (W = u[N], c.hasOwnProperty(N) && (B != null || W != null))
            switch (N) {
              case "type":
                s = B;
                break;
              case "name":
                r = B;
                break;
              case "checked":
                C = B;
                break;
              case "defaultChecked":
                Z = B;
                break;
              case "value":
                y = B;
                break;
              case "defaultValue":
                p = B;
                break;
              case "children":
              case "dangerouslySetInnerHTML":
                if (B != null)
                  throw Error(_(137, n));
                break;
              default:
                B !== W && qe(
                  l,
                  n,
                  N,
                  B,
                  c,
                  W
                );
            }
        }
        ys(
          l,
          y,
          p,
          S,
          C,
          Z,
          s,
          r
        );
        return;
      case "select":
        B = y = p = N = null;
        for (s in u)
          if (S = u[s], u.hasOwnProperty(s) && S != null)
            switch (s) {
              case "value":
                break;
              case "multiple":
                B = S;
              default:
                c.hasOwnProperty(s) || qe(
                  l,
                  n,
                  s,
                  null,
                  c,
                  S
                );
            }
        for (r in c)
          if (s = c[r], S = u[r], c.hasOwnProperty(r) && (s != null || S != null))
            switch (r) {
              case "value":
                N = s;
                break;
              case "defaultValue":
                p = s;
                break;
              case "multiple":
                y = s;
              default:
                s !== S && qe(
                  l,
                  n,
                  r,
                  s,
                  c,
                  S
                );
            }
        n = p, u = y, c = B, N != null ? Fi(l, !!u, N, !1) : !!c != !!u && (n != null ? Fi(l, !!u, n, !0) : Fi(l, !!u, u ? [] : "", !1));
        return;
      case "textarea":
        B = N = null;
        for (p in u)
          if (r = u[p], u.hasOwnProperty(p) && r != null && !c.hasOwnProperty(p))
            switch (p) {
              case "value":
                break;
              case "children":
                break;
              default:
                qe(l, n, p, null, c, r);
            }
        for (y in c)
          if (r = c[y], s = u[y], c.hasOwnProperty(y) && (r != null || s != null))
            switch (y) {
              case "value":
                N = r;
                break;
              case "defaultValue":
                B = r;
                break;
              case "children":
                break;
              case "dangerouslySetInnerHTML":
                if (r != null) throw Error(_(91));
                break;
              default:
                r !== s && qe(l, n, y, r, c, s);
            }
        Yh(l, N, B);
        return;
      case "option":
        for (var Te in u)
          if (N = u[Te], u.hasOwnProperty(Te) && N != null && !c.hasOwnProperty(Te))
            switch (Te) {
              case "selected":
                l.selected = !1;
                break;
              default:
                qe(
                  l,
                  n,
                  Te,
                  null,
                  c,
                  N
                );
            }
        for (S in c)
          if (N = c[S], B = u[S], c.hasOwnProperty(S) && N !== B && (N != null || B != null))
            switch (S) {
              case "selected":
                l.selected = N && typeof N != "function" && typeof N != "symbol";
                break;
              default:
                qe(
                  l,
                  n,
                  S,
                  N,
                  c,
                  B
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
        for (var Ee in u)
          N = u[Ee], u.hasOwnProperty(Ee) && N != null && !c.hasOwnProperty(Ee) && qe(l, n, Ee, null, c, N);
        for (C in c)
          if (N = c[C], B = u[C], c.hasOwnProperty(C) && N !== B && (N != null || B != null))
            switch (C) {
              case "children":
              case "dangerouslySetInnerHTML":
                if (N != null)
                  throw Error(_(137, n));
                break;
              default:
                qe(
                  l,
                  n,
                  C,
                  N,
                  c,
                  B
                );
            }
        return;
      default:
        if (Ii(n)) {
          for (var yt in u)
            N = u[yt], u.hasOwnProperty(yt) && N !== void 0 && !c.hasOwnProperty(yt) && Y(
              l,
              n,
              yt,
              void 0,
              c,
              N
            );
          for (Z in c)
            N = c[Z], B = u[Z], !c.hasOwnProperty(Z) || N === B || N === void 0 && B === void 0 || Y(
              l,
              n,
              Z,
              N,
              c,
              B
            );
          return;
        }
    }
    for (var M in u)
      N = u[M], u.hasOwnProperty(M) && N != null && !c.hasOwnProperty(M) && qe(l, n, M, null, c, N);
    for (W in c)
      N = c[W], B = u[W], !c.hasOwnProperty(W) || N === B || N == null && B == null || qe(l, n, W, N, c, B);
  }
  var Ur = null, Cr = null;
  function en(l) {
    return l.nodeType === 9 ? l : l.ownerDocument;
  }
  function Vu(l) {
    switch (l) {
      case "http://www.w3.org/2000/svg":
        return 1;
      case "http://www.w3.org/1998/Math/MathML":
        return 2;
      default:
        return 0;
    }
  }
  function $o(l, n) {
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
  function uu(l, n) {
    return l === "textarea" || l === "noscript" || typeof n.children == "string" || typeof n.children == "number" || typeof n.children == "bigint" || typeof n.dangerouslySetInnerHTML == "object" && n.dangerouslySetInnerHTML !== null && n.dangerouslySetInnerHTML.__html != null;
  }
  var Wo = null;
  function Xu() {
    var l = window.event;
    return l && l.type === "popstate" ? l === Wo ? !1 : (Wo = l, !0) : (Wo = null, !1);
  }
  var Vd = typeof setTimeout == "function" ? setTimeout : void 0, Bg = typeof clearTimeout == "function" ? clearTimeout : void 0, Av = typeof Promise == "function" ? Promise : void 0, Yg = typeof queueMicrotask == "function" ? queueMicrotask : typeof Av < "u" ? function(l) {
    return Av.resolve(null).then(l).catch(iu);
  } : Vd;
  function iu(l) {
    setTimeout(function() {
      throw l;
    });
  }
  function Hi(l) {
    return l === "head";
  }
  function Xd(l, n) {
    var u = n, c = 0, r = 0;
    do {
      var s = u.nextSibling;
      if (l.removeChild(u), s && s.nodeType === 8)
        if (u = s.data, u === "/$") {
          if (0 < c && 8 > c) {
            u = c;
            var y = l.ownerDocument;
            if (u & 1 && ga(y.documentElement), u & 2 && ga(y.body), u & 4)
              for (u = y.head, ga(u), y = u.firstChild; y; ) {
                var p = y.nextSibling, S = y.nodeName;
                y[he] || S === "SCRIPT" || S === "STYLE" || S === "LINK" && y.rel.toLowerCase() === "stylesheet" || u.removeChild(y), y = p;
              }
          }
          if (r === 0) {
            l.removeChild(s), ou(n);
            return;
          }
          r--;
        } else
          u === "$" || u === "$?" || u === "$!" ? r++ : c = u.charCodeAt(0) - 48;
      else c = 0;
      u = s;
    } while (u);
    ou(n);
  }
  function xr(l) {
    var n = l.firstChild;
    for (n && n.nodeType === 10 && (n = n.nextSibling); n; ) {
      var u = n;
      switch (n = n.nextSibling, u.nodeName) {
        case "HTML":
        case "HEAD":
        case "BODY":
          xr(u), Rf(u);
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
  function Fo(l, n, u, c) {
    for (; l.nodeType === 1; ) {
      var r = u;
      if (l.nodeName.toLowerCase() !== n.toLowerCase()) {
        if (!c && (l.nodeName !== "INPUT" || l.type !== "hidden"))
          break;
      } else if (c) {
        if (!l[he])
          switch (n) {
            case "meta":
              if (!l.hasAttribute("itemprop")) break;
              return l;
            case "link":
              if (s = l.getAttribute("rel"), s === "stylesheet" && l.hasAttribute("data-precedence"))
                break;
              if (s !== r.rel || l.getAttribute("href") !== (r.href == null || r.href === "" ? null : r.href) || l.getAttribute("crossorigin") !== (r.crossOrigin == null ? null : r.crossOrigin) || l.getAttribute("title") !== (r.title == null ? null : r.title))
                break;
              return l;
            case "style":
              if (l.hasAttribute("data-precedence")) break;
              return l;
            case "script":
              if (s = l.getAttribute("src"), (s !== (r.src == null ? null : r.src) || l.getAttribute("type") !== (r.type == null ? null : r.type) || l.getAttribute("crossorigin") !== (r.crossOrigin == null ? null : r.crossOrigin)) && s && l.hasAttribute("async") && !l.hasAttribute("itemprop"))
                break;
              return l;
            default:
              return l;
          }
      } else if (n === "input" && l.type === "hidden") {
        var s = r.name == null ? null : "" + r.name;
        if (r.type === "hidden" && l.getAttribute("name") === s)
          return l;
      } else return l;
      if (l = En(l.nextSibling), l === null) break;
    }
    return null;
  }
  function jg(l, n, u) {
    if (n === "") return null;
    for (; l.nodeType !== 3; )
      if ((l.nodeType !== 1 || l.nodeName !== "INPUT" || l.type !== "hidden") && !u || (l = En(l.nextSibling), l === null)) return null;
    return l;
  }
  function Hr(l) {
    return l.data === "$!" || l.data === "$?" && l.ownerDocument.readyState === "complete";
  }
  function Gg(l, n) {
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
  function En(l) {
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
  var Ni = null;
  function ql(l) {
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
    switch (n = en(u), l) {
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
  function ga(l) {
    for (var n = l.attributes; n.length; )
      l.removeAttributeNode(n[0]);
    Rf(l);
  }
  var It = /* @__PURE__ */ new Map(), Zl = /* @__PURE__ */ new Set();
  function Qd(l) {
    return typeof l.getRootNode == "function" ? l.getRootNode() : l.nodeType === 9 ? l : l.ownerDocument;
  }
  var Qu = X.d;
  X.d = {
    f: Zd,
    r: Kd,
    D: Zu,
    C: Jd,
    L: wi,
    m: Kl,
    X: qi,
    S: ba,
    M: Om
  };
  function Zd() {
    var l = Qu.f(), n = Cc();
    return l || n;
  }
  function Kd(l) {
    var n = Ji(l);
    n !== null && n.tag === 5 && n.type === "form" ? _o(n) : Qu.r(l);
  }
  var Bl = typeof document > "u" ? null : document;
  function Rn(l, n, u) {
    var c = Bl;
    if (c && typeof n == "string" && n) {
      var r = Ga(n);
      r = 'link[rel="' + l + '"][href="' + r + '"]', typeof u == "string" && (r += '[crossorigin="' + u + '"]'), Zl.has(r) || (Zl.add(r), l = { rel: l, crossOrigin: u, href: n }, c.querySelector(r) === null && (n = c.createElement("link"), Ue(n, "link", l), fl(n), c.head.appendChild(n)));
    }
  }
  function Zu(l) {
    Qu.D(l), Rn("dns-prefetch", l, null);
  }
  function Jd(l, n) {
    Qu.C(l, n), Rn("preconnect", l, n);
  }
  function wi(l, n, u) {
    Qu.L(l, n, u);
    var c = Bl;
    if (c && l && n) {
      var r = 'link[rel="preload"][as="' + Ga(n) + '"]';
      n === "image" && u && u.imageSrcSet ? (r += '[imagesrcset="' + Ga(
        u.imageSrcSet
      ) + '"]', typeof u.imageSizes == "string" && (r += '[imagesizes="' + Ga(
        u.imageSizes
      ) + '"]')) : r += '[href="' + Ga(l) + '"]';
      var s = r;
      switch (n) {
        case "style":
          s = Io(l);
          break;
        case "script":
          s = tn(l);
      }
      It.has(s) || (l = ie(
        {
          rel: "preload",
          href: n === "image" && u && u.imageSrcSet ? void 0 : l,
          as: n
        },
        u
      ), It.set(s, l), c.querySelector(r) !== null || n === "style" && c.querySelector(Po(s)) || n === "script" && c.querySelector(qc(s)) || (n = c.createElement("link"), Ue(n, "link", l), fl(n), c.head.appendChild(n)));
    }
  }
  function Kl(l, n) {
    Qu.m(l, n);
    var u = Bl;
    if (u && l) {
      var c = n && typeof n.as == "string" ? n.as : "script", r = 'link[rel="modulepreload"][as="' + Ga(c) + '"][href="' + Ga(l) + '"]', s = r;
      switch (c) {
        case "audioworklet":
        case "paintworklet":
        case "serviceworker":
        case "sharedworker":
        case "worker":
        case "script":
          s = tn(l);
      }
      if (!It.has(s) && (l = ie({ rel: "modulepreload", href: l }, n), It.set(s, l), u.querySelector(r) === null)) {
        switch (c) {
          case "audioworklet":
          case "paintworklet":
          case "serviceworker":
          case "sharedworker":
          case "worker":
          case "script":
            if (u.querySelector(qc(s)))
              return;
        }
        c = u.createElement("link"), Ue(c, "link", l), fl(c), u.head.appendChild(c);
      }
    }
  }
  function ba(l, n, u) {
    Qu.S(l, n, u);
    var c = Bl;
    if (c && l) {
      var r = Tu(c).hoistableStyles, s = Io(l);
      n = n || "default";
      var y = r.get(s);
      if (!y) {
        var p = { loading: 0, preload: null };
        if (y = c.querySelector(
          Po(s)
        ))
          p.loading = 5;
        else {
          l = ie(
            { rel: "stylesheet", href: l, "data-precedence": n },
            u
          ), (u = It.get(s)) && $d(l, u);
          var S = y = c.createElement("link");
          fl(S), Ue(S, "link", l), S._p = new Promise(function(C, Z) {
            S.onload = C, S.onerror = Z;
          }), S.addEventListener("load", function() {
            p.loading |= 1;
          }), S.addEventListener("error", function() {
            p.loading |= 2;
          }), p.loading |= 4, kd(y, n, c);
        }
        y = {
          type: "stylesheet",
          instance: y,
          count: 1,
          state: p
        }, r.set(s, y);
      }
    }
  }
  function qi(l, n) {
    Qu.X(l, n);
    var u = Bl;
    if (u && l) {
      var c = Tu(u).hoistableScripts, r = tn(l), s = c.get(r);
      s || (s = u.querySelector(qc(r)), s || (l = ie({ src: l, async: !0 }, n), (n = It.get(r)) && Wd(l, n), s = u.createElement("script"), fl(s), Ue(s, "link", l), u.head.appendChild(s)), s = {
        type: "script",
        instance: s,
        count: 1,
        state: null
      }, c.set(r, s));
    }
  }
  function Om(l, n) {
    Qu.M(l, n);
    var u = Bl;
    if (u && l) {
      var c = Tu(u).hoistableScripts, r = tn(l), s = c.get(r);
      s || (s = u.querySelector(qc(r)), s || (l = ie({ src: l, async: !0, type: "module" }, n), (n = It.get(r)) && Wd(l, n), s = u.createElement("script"), fl(s), Ue(s, "link", l), u.head.appendChild(s)), s = {
        type: "script",
        instance: s,
        count: 1,
        state: null
      }, c.set(r, s));
    }
  }
  function Ov(l, n, u, c) {
    var r = (r = oe.current) ? Qd(r) : null;
    if (!r) throw Error(_(446));
    switch (l) {
      case "meta":
      case "title":
        return null;
      case "style":
        return typeof u.precedence == "string" && typeof u.href == "string" ? (n = Io(u.href), u = Tu(
          r
        ).hoistableStyles, c = u.get(n), c || (c = {
          type: "style",
          instance: null,
          count: 0,
          state: null
        }, u.set(n, c)), c) : { type: "void", instance: null, count: 0, state: null };
      case "link":
        if (u.rel === "stylesheet" && typeof u.href == "string" && typeof u.precedence == "string") {
          l = Io(u.href);
          var s = Tu(
            r
          ).hoistableStyles, y = s.get(l);
          if (y || (r = r.ownerDocument || r, y = {
            type: "stylesheet",
            instance: null,
            count: 0,
            state: { loading: 0, preload: null }
          }, s.set(l, y), (s = r.querySelector(
            Po(l)
          )) && !s._p && (y.instance = s, y.state.loading = 5), It.has(l) || (u = {
            rel: "preload",
            as: "style",
            href: u.href,
            crossOrigin: u.crossOrigin,
            integrity: u.integrity,
            media: u.media,
            hrefLang: u.hrefLang,
            referrerPolicy: u.referrerPolicy
          }, It.set(l, u), s || Dv(
            r,
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
        return n = u.async, u = u.src, typeof u == "string" && n && typeof n != "function" && typeof n != "symbol" ? (n = tn(u), u = Tu(
          r
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
  function Io(l) {
    return 'href="' + Ga(l) + '"';
  }
  function Po(l) {
    return 'link[rel="stylesheet"][' + l + "]";
  }
  function ef(l) {
    return ie({}, l, {
      "data-precedence": l.precedence,
      precedence: null
    });
  }
  function Dv(l, n, u, c) {
    l.querySelector('link[rel="preload"][as="style"][' + n + "]") ? c.loading = 1 : (n = l.createElement("link"), c.preload = n, n.addEventListener("load", function() {
      return c.loading |= 1;
    }), n.addEventListener("error", function() {
      return c.loading |= 2;
    }), Ue(n, "link", u), fl(n), l.head.appendChild(n));
  }
  function tn(l) {
    return '[src="' + Ga(l) + '"]';
  }
  function qc(l) {
    return "script[async]" + l;
  }
  function zv(l, n, u) {
    if (n.count++, n.instance === null)
      switch (n.type) {
        case "style":
          var c = l.querySelector(
            'style[data-href~="' + Ga(u.href) + '"]'
          );
          if (c)
            return n.instance = c, fl(c), c;
          var r = ie({}, u, {
            "data-href": u.href,
            "data-precedence": u.precedence,
            href: null,
            precedence: null
          });
          return c = (l.ownerDocument || l).createElement(
            "style"
          ), fl(c), Ue(c, "style", r), kd(c, u.precedence, l), n.instance = c;
        case "stylesheet":
          r = Io(u.href);
          var s = l.querySelector(
            Po(r)
          );
          if (s)
            return n.state.loading |= 4, n.instance = s, fl(s), s;
          c = ef(u), (r = It.get(r)) && $d(c, r), s = (l.ownerDocument || l).createElement("link"), fl(s);
          var y = s;
          return y._p = new Promise(function(p, S) {
            y.onload = p, y.onerror = S;
          }), Ue(s, "link", c), n.state.loading |= 4, kd(s, u.precedence, l), n.instance = s;
        case "script":
          return s = tn(u.src), (r = l.querySelector(
            qc(s)
          )) ? (n.instance = r, fl(r), r) : (c = u, (r = It.get(s)) && (c = ie({}, u), Wd(c, r)), l = l.ownerDocument || l, r = l.createElement("script"), fl(r), Ue(r, "link", c), l.head.appendChild(r), n.instance = r);
        case "void":
          return null;
        default:
          throw Error(_(443, n.type));
      }
    else
      n.type === "stylesheet" && (n.state.loading & 4) === 0 && (c = n.instance, n.state.loading |= 4, kd(c, u.precedence, l));
    return n.instance;
  }
  function kd(l, n, u) {
    for (var c = u.querySelectorAll(
      'link[rel="stylesheet"][data-precedence],style[data-precedence]'
    ), r = c.length ? c[c.length - 1] : null, s = r, y = 0; y < c.length; y++) {
      var p = c[y];
      if (p.dataset.precedence === n) s = p;
      else if (s !== r) break;
    }
    s ? s.parentNode.insertBefore(l, s.nextSibling) : (n = u.nodeType === 9 ? u.head : u, n.insertBefore(l, n.firstChild));
  }
  function $d(l, n) {
    l.crossOrigin == null && (l.crossOrigin = n.crossOrigin), l.referrerPolicy == null && (l.referrerPolicy = n.referrerPolicy), l.title == null && (l.title = n.title);
  }
  function Wd(l, n) {
    l.crossOrigin == null && (l.crossOrigin = n.crossOrigin), l.referrerPolicy == null && (l.referrerPolicy = n.referrerPolicy), l.integrity == null && (l.integrity = n.integrity);
  }
  var Bi = null;
  function Dm(l, n, u) {
    if (Bi === null) {
      var c = /* @__PURE__ */ new Map(), r = Bi = /* @__PURE__ */ new Map();
      r.set(u, c);
    } else
      r = Bi, c = r.get(u), c || (c = /* @__PURE__ */ new Map(), r.set(u, c));
    if (c.has(l)) return c;
    for (c.set(l, null), u = u.getElementsByTagName(l), r = 0; r < u.length; r++) {
      var s = u[r];
      if (!(s[he] || s[gl] || l === "link" && s.getAttribute("rel") === "stylesheet") && s.namespaceURI !== "http://www.w3.org/2000/svg") {
        var y = s.getAttribute(n) || "";
        y = l + y;
        var p = c.get(y);
        p ? p.push(s) : c.set(y, [s]);
      }
    }
    return c;
  }
  function zm(l, n, u) {
    l = l.ownerDocument || l, l.head.insertBefore(
      u,
      n === "title" ? l.querySelector("head > title") : null
    );
  }
  function Mv(l, n, u) {
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
  function Mm(l) {
    return !(l.type === "stylesheet" && (l.state.loading & 3) === 0);
  }
  var tf = null;
  function _v() {
  }
  function Uv(l, n, u) {
    if (tf === null) throw Error(_(475));
    var c = tf;
    if (n.type === "stylesheet" && (typeof u.media != "string" || matchMedia(u.media).matches !== !1) && (n.state.loading & 4) === 0) {
      if (n.instance === null) {
        var r = Io(u.href), s = l.querySelector(
          Po(r)
        );
        if (s) {
          l = s._p, l !== null && typeof l == "object" && typeof l.then == "function" && (c.count++, c = Nr.bind(c), l.then(c, c)), n.state.loading |= 4, n.instance = s, fl(s);
          return;
        }
        s = l.ownerDocument || l, u = ef(u), (r = It.get(r)) && $d(u, r), s = s.createElement("link"), fl(s);
        var y = s;
        y._p = new Promise(function(p, S) {
          y.onload = p, y.onerror = S;
        }), Ue(s, "link", u), n.instance = s;
      }
      c.stylesheets === null && (c.stylesheets = /* @__PURE__ */ new Map()), c.stylesheets.set(n, l), (l = n.state.preload) && (n.state.loading & 3) === 0 && (c.count++, n = Nr.bind(c), l.addEventListener("load", n), l.addEventListener("error", n));
    }
  }
  function _m() {
    if (tf === null) throw Error(_(475));
    var l = tf;
    return l.stylesheets && l.count === 0 && wr(l, l.stylesheets), 0 < l.count ? function(n) {
      var u = setTimeout(function() {
        if (l.stylesheets && wr(l, l.stylesheets), l.unsuspend) {
          var c = l.unsuspend;
          l.unsuspend = null, c();
        }
      }, 6e4);
      return l.unsuspend = n, function() {
        l.unsuspend = null, clearTimeout(u);
      };
    } : null;
  }
  function Nr() {
    if (this.count--, this.count === 0) {
      if (this.stylesheets) wr(this, this.stylesheets);
      else if (this.unsuspend) {
        var l = this.unsuspend;
        this.unsuspend = null, l();
      }
    }
  }
  var lf = null;
  function wr(l, n) {
    l.stylesheets = null, l.unsuspend !== null && (l.count++, lf = /* @__PURE__ */ new Map(), n.forEach(Ha, l), lf = null, Nr.call(l));
  }
  function Ha(l, n) {
    if (!(n.state.loading & 4)) {
      var u = lf.get(l);
      if (u) var c = u.get(null);
      else {
        u = /* @__PURE__ */ new Map(), lf.set(l, u);
        for (var r = l.querySelectorAll(
          "link[data-precedence],style[data-precedence]"
        ), s = 0; s < r.length; s++) {
          var y = r[s];
          (y.nodeName === "LINK" || y.getAttribute("media") !== "not all") && (u.set(y.dataset.precedence, y), c = y);
        }
        c && u.set(null, c);
      }
      r = n.instance, y = r.getAttribute("data-precedence"), s = u.get(y) || c, s === c && u.set(null, r), u.set(y, r), this.count++, c = Nr.bind(this), r.addEventListener("load", c), r.addEventListener("error", c), s ? s.parentNode.insertBefore(r, s.nextSibling) : (l = l.nodeType === 9 ? l.head : l, l.insertBefore(r, l.firstChild)), n.state.loading |= 4;
    }
  }
  var Sa = {
    $$typeof: Ke,
    Provider: null,
    Consumer: null,
    _currentValue: I,
    _currentValue2: I,
    _threadCount: 0
  };
  function Lg(l, n, u, c, r, s, y, p) {
    this.tag = 1, this.containerInfo = l, this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.next = this.pendingContext = this.context = this.cancelPendingCommit = null, this.callbackPriority = 0, this.expirationTimes = pe(-1), this.entangledLanes = this.shellSuspendCounter = this.errorRecoveryDisabledLanes = this.expiredLanes = this.warmLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = pe(0), this.hiddenUpdates = pe(null), this.identifierPrefix = c, this.onUncaughtError = r, this.onCaughtError = s, this.onRecoverableError = y, this.pooledCache = null, this.pooledCacheLanes = 0, this.formState = p, this.incompleteTransitions = /* @__PURE__ */ new Map();
  }
  function Um(l, n, u, c, r, s, y, p, S, C, Z, W) {
    return l = new Lg(
      l,
      n,
      u,
      y,
      p,
      S,
      C,
      W
    ), n = 1, s === !0 && (n |= 24), s = fa(3, null, null, n), l.current = s, s.stateNode = l, n = Ao(), n.refCount++, l.pooledCache = n, n.refCount++, s.memoizedState = {
      element: c,
      isDehydrated: u,
      cache: n
    }, Ls(s), l;
  }
  function Cm(l) {
    return l ? (l = po, l) : po;
  }
  function xm(l, n, u, c, r, s) {
    r = Cm(r), c.context === null ? c.context = r : c.pendingContext = r, c = sa(n), c.payload = { element: u }, s = s === void 0 ? null : s, s !== null && (c.callback = s), u = Qn(l, c, n), u !== null && (Ca(u, l, n), mc(u, l, n));
  }
  function Hm(l, n) {
    if (l = l.memoizedState, l !== null && l.dehydrated !== null) {
      var u = l.retryLane;
      l.retryLane = u !== 0 && u < n ? u : n;
    }
  }
  function Fd(l, n) {
    Hm(l, n), (l = l.alternate) && Hm(l, n);
  }
  function Nm(l) {
    if (l.tag === 13) {
      var n = Yn(l, 67108864);
      n !== null && Ca(n, l, 67108864), Fd(l, 67108864);
    }
  }
  var qr = !0;
  function Cv(l, n, u, c) {
    var r = R.T;
    R.T = null;
    var s = X.p;
    try {
      X.p = 2, wm(l, n, u, c);
    } finally {
      X.p = s, R.T = r;
    }
  }
  function xv(l, n, u, c) {
    var r = R.T;
    R.T = null;
    var s = X.p;
    try {
      X.p = 8, wm(l, n, u, c);
    } finally {
      X.p = s, R.T = r;
    }
  }
  function wm(l, n, u, c) {
    if (qr) {
      var r = Id(c);
      if (r === null)
        Pa(
          l,
          n,
          c,
          Pd,
          u
        ), Bc(l, c);
      else if (Nv(
        r,
        l,
        n,
        u,
        c
      ))
        c.stopPropagation();
      else if (Bc(l, c), n & 4 && -1 < Hv.indexOf(l)) {
        for (; r !== null; ) {
          var s = Ji(r);
          if (s !== null)
            switch (s.tag) {
              case 3:
                if (s = s.stateNode, s.current.memoizedState.isDehydrated) {
                  var y = Ml(s.pendingLanes);
                  if (y !== 0) {
                    var p = s;
                    for (p.pendingLanes |= 2, p.entangledLanes |= 2; y; ) {
                      var S = 1 << 31 - zl(y);
                      p.entanglements[1] |= S, y &= ~S;
                    }
                    va(s), (gt & 6) === 0 && (_d = vl() + 500, Rr(0));
                  }
                }
                break;
              case 13:
                p = Yn(s, 2), p !== null && Ca(p, s, 2), Cc(), Fd(s, 2);
            }
          if (s = Id(c), s === null && Pa(
            l,
            n,
            c,
            Pd,
            u
          ), s === r) break;
          r = s;
        }
        r !== null && c.stopPropagation();
      } else
        Pa(
          l,
          n,
          c,
          null,
          u
        );
    }
  }
  function Id(l) {
    return l = vs(l), qm(l);
  }
  var Pd = null;
  function qm(l) {
    if (Pd = null, l = _l(l), l !== null) {
      var n = Ae(l);
      if (n === null) l = null;
      else {
        var u = n.tag;
        if (u === 13) {
          if (l = Ne(n), l !== null) return l;
          l = null;
        } else if (u === 3) {
          if (n.stateNode.current.memoizedState.isDehydrated)
            return n.tag === 3 ? n.stateNode.containerInfo : null;
          l = null;
        } else n !== l && (l = null);
      }
    }
    return Pd = l, null;
  }
  function Bm(l) {
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
        switch (Pu()) {
          case cs:
            return 2;
          case Je:
            return 8;
          case Un:
          case to:
            return 32;
          case Su:
            return 268435456;
          default:
            return 32;
        }
      default:
        return 32;
    }
  }
  var af = !1, cu = null, Ku = null, Ju = null, Br = /* @__PURE__ */ new Map(), Yr = /* @__PURE__ */ new Map(), Yi = [], Hv = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(
    " "
  );
  function Bc(l, n) {
    switch (l) {
      case "focusin":
      case "focusout":
        cu = null;
        break;
      case "dragenter":
      case "dragleave":
        Ku = null;
        break;
      case "mouseover":
      case "mouseout":
        Ju = null;
        break;
      case "pointerover":
      case "pointerout":
        Br.delete(n.pointerId);
        break;
      case "gotpointercapture":
      case "lostpointercapture":
        Yr.delete(n.pointerId);
    }
  }
  function Yc(l, n, u, c, r, s) {
    return l === null || l.nativeEvent !== s ? (l = {
      blockedOn: n,
      domEventName: u,
      eventSystemFlags: c,
      nativeEvent: s,
      targetContainers: [r]
    }, n !== null && (n = Ji(n), n !== null && Nm(n)), l) : (l.eventSystemFlags |= c, n = l.targetContainers, r !== null && n.indexOf(r) === -1 && n.push(r), l);
  }
  function Nv(l, n, u, c, r) {
    switch (n) {
      case "focusin":
        return cu = Yc(
          cu,
          l,
          n,
          u,
          c,
          r
        ), !0;
      case "dragenter":
        return Ku = Yc(
          Ku,
          l,
          n,
          u,
          c,
          r
        ), !0;
      case "mouseover":
        return Ju = Yc(
          Ju,
          l,
          n,
          u,
          c,
          r
        ), !0;
      case "pointerover":
        var s = r.pointerId;
        return Br.set(
          s,
          Yc(
            Br.get(s) || null,
            l,
            n,
            u,
            c,
            r
          )
        ), !0;
      case "gotpointercapture":
        return s = r.pointerId, Yr.set(
          s,
          Yc(
            Yr.get(s) || null,
            l,
            n,
            u,
            c,
            r
          )
        ), !0;
    }
    return !1;
  }
  function Ym(l) {
    var n = _l(l.target);
    if (n !== null) {
      var u = Ae(n);
      if (u !== null) {
        if (n = u.tag, n === 13) {
          if (n = Ne(u), n !== null) {
            l.blockedOn = n, Hh(l.priority, function() {
              if (u.tag === 13) {
                var c = Ua();
                c = al(c);
                var r = Yn(u, c);
                r !== null && Ca(r, u, c), Fd(u, c);
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
  function jr(l) {
    if (l.blockedOn !== null) return !1;
    for (var n = l.targetContainers; 0 < n.length; ) {
      var u = Id(l.nativeEvent);
      if (u === null) {
        u = l.nativeEvent;
        var c = new u.constructor(
          u.type,
          u
        );
        Pi = c, u.target.dispatchEvent(c), Pi = null;
      } else
        return n = Ji(u), n !== null && Nm(n), l.blockedOn = u, !1;
      n.shift();
    }
    return !0;
  }
  function Gr(l, n, u) {
    jr(l) && u.delete(n);
  }
  function nf() {
    af = !1, cu !== null && jr(cu) && (cu = null), Ku !== null && jr(Ku) && (Ku = null), Ju !== null && jr(Ju) && (Ju = null), Br.forEach(Gr), Yr.forEach(Gr);
  }
  function eh(l, n) {
    l.blockedOn === n && (l.blockedOn = null, af || (af = !0, H.unstable_scheduleCallback(
      H.unstable_NormalPriority,
      nf
    )));
  }
  var jc = null;
  function jm(l) {
    jc !== l && (jc = l, H.unstable_scheduleCallback(
      H.unstable_NormalPriority,
      function() {
        jc === l && (jc = null);
        for (var n = 0; n < l.length; n += 3) {
          var u = l[n], c = l[n + 1], r = l[n + 2];
          if (typeof c != "function") {
            if (qm(c || u) === null)
              continue;
            break;
          }
          var s = Ji(u);
          s !== null && (l.splice(n, 3), n -= 3, cd(
            s,
            {
              pending: !0,
              data: r,
              method: u.method,
              action: c
            },
            c,
            r
          ));
        }
      }
    ));
  }
  function ou(l) {
    function n(S) {
      return eh(S, l);
    }
    cu !== null && eh(cu, l), Ku !== null && eh(Ku, l), Ju !== null && eh(Ju, l), Br.forEach(n), Yr.forEach(n);
    for (var u = 0; u < Yi.length; u++) {
      var c = Yi[u];
      c.blockedOn === l && (c.blockedOn = null);
    }
    for (; 0 < Yi.length && (u = Yi[0], u.blockedOn === null); )
      Ym(u), u.blockedOn === null && Yi.shift();
    if (u = (l.ownerDocument || l).$$reactFormReplay, u != null)
      for (c = 0; c < u.length; c += 3) {
        var r = u[c], s = u[c + 1], y = r[$l] || null;
        if (typeof s == "function")
          y || jm(u);
        else if (y) {
          var p = null;
          if (s && s.hasAttribute("formAction")) {
            if (r = s, y = s[$l] || null)
              p = y.formAction;
            else if (qm(r) !== null) continue;
          } else p = y.action;
          typeof p == "function" ? u[c + 1] = p : (u.splice(c, 3), c -= 3), jm(u);
        }
      }
  }
  function Gm(l) {
    this._internalRoot = l;
  }
  th.prototype.render = Gm.prototype.render = function(l) {
    var n = this._internalRoot;
    if (n === null) throw Error(_(409));
    var u = n.current, c = Ua();
    xm(u, c, l, n, null, null);
  }, th.prototype.unmount = Gm.prototype.unmount = function() {
    var l = this._internalRoot;
    if (l !== null) {
      this._internalRoot = null;
      var n = l.containerInfo;
      xm(l.current, 2, null, l, null, null), Cc(), n[no] = null;
    }
  };
  function th(l) {
    this._internalRoot = l;
  }
  th.prototype.unstable_scheduleHydration = function(l) {
    if (l) {
      var n = fs();
      l = { blockedOn: null, target: l, priority: n };
      for (var u = 0; u < Yi.length && n !== 0 && n < Yi[u].priority; u++) ;
      Yi.splice(u, 0, l), u === 0 && Ym(l);
    }
  };
  var Lm = F.version;
  if (Lm !== "19.1.1")
    throw Error(
      _(
        527,
        Lm,
        "19.1.1"
      )
    );
  X.findDOMNode = function(l) {
    var n = l._reactInternals;
    if (n === void 0)
      throw typeof l.render == "function" ? Error(_(188)) : (l = Object.keys(l).join(","), Error(_(268, l)));
    return l = j(n), l = l !== null ? k(l) : null, l = l === null ? null : l.stateNode, l;
  };
  var ta = {
    bundleType: 0,
    version: "19.1.1",
    rendererPackageName: "react-dom",
    currentDispatcherRef: R,
    reconcilerVersion: "19.1.1"
  };
  if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u") {
    var Lr = __REACT_DEVTOOLS_GLOBAL_HOOK__;
    if (!Lr.isDisabled && Lr.supportsFiber)
      try {
        ei = Lr.inject(
          ta
        ), Dl = Lr;
      } catch {
      }
  }
  return Ap.createRoot = function(l, n) {
    if (!re(l)) throw Error(_(299));
    var u = !1, c = "", r = xo, s = Yy, y = dr, p = null;
    return n != null && (n.unstable_strictMode === !0 && (u = !0), n.identifierPrefix !== void 0 && (c = n.identifierPrefix), n.onUncaughtError !== void 0 && (r = n.onUncaughtError), n.onCaughtError !== void 0 && (s = n.onCaughtError), n.onRecoverableError !== void 0 && (y = n.onRecoverableError), n.unstable_transitionCallbacks !== void 0 && (p = n.unstable_transitionCallbacks)), n = Um(
      l,
      1,
      !1,
      null,
      null,
      u,
      c,
      r,
      s,
      y,
      p,
      null
    ), l[no] = n.current, Em(l), new Gm(n);
  }, Ap.hydrateRoot = function(l, n, u) {
    if (!re(l)) throw Error(_(299));
    var c = !1, r = "", s = xo, y = Yy, p = dr, S = null, C = null;
    return u != null && (u.unstable_strictMode === !0 && (c = !0), u.identifierPrefix !== void 0 && (r = u.identifierPrefix), u.onUncaughtError !== void 0 && (s = u.onUncaughtError), u.onCaughtError !== void 0 && (y = u.onCaughtError), u.onRecoverableError !== void 0 && (p = u.onRecoverableError), u.unstable_transitionCallbacks !== void 0 && (S = u.unstable_transitionCallbacks), u.formState !== void 0 && (C = u.formState)), n = Um(
      l,
      1,
      !0,
      n,
      u ?? null,
      c,
      r,
      s,
      y,
      p,
      S,
      C
    ), n.context = Cm(null), u = n.current, c = Ua(), c = al(c), r = sa(c), r.callback = null, Qn(u, r, c), u = c, n.current.lanes = u, we(n, u), va(n), l[no] = n.current, Em(l), new th(n);
  }, Ap.version = "19.1.1", Ap;
}
var Op = {}, tS;
function _T() {
  return tS || (tS = 1, Pt.env.NODE_ENV !== "production" && function() {
    function H(e, t) {
      for (e = e.memoizedState; e !== null && 0 < t; )
        e = e.next, t--;
      return e;
    }
    function F(e, t, a, i) {
      if (a >= t.length) return i;
      var o = t[a], f = qe(e) ? e.slice() : ke({}, e);
      return f[o] = F(e[o], t, a + 1, i), f;
    }
    function Re(e, t, a) {
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
      var o = t[i], f = qe(e) ? e.slice() : ke({}, e);
      return i + 1 === t.length ? (f[a[i]] = f[o], qe(f) ? f.splice(o, 1) : delete f[o]) : f[o] = _(
        e[o],
        t,
        a,
        i + 1
      ), f;
    }
    function re(e, t, a) {
      var i = t[a], o = qe(e) ? e.slice() : ke({}, e);
      return a + 1 === t.length ? (qe(o) ? o.splice(i, 1) : delete o[i], o) : (o[i] = re(e[i], t, a + 1), o);
    }
    function Ae() {
      return !1;
    }
    function Ne() {
      return null;
    }
    function st() {
    }
    function j() {
      console.error(
        "Do not call Hooks inside useEffect(...), useMemo(...), or other built-in Hooks. You can only call Hooks at the top level of your React function. For more information, see https://react.dev/link/rules-of-hooks"
      );
    }
    function k() {
      console.error(
        "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
      );
    }
    function ie() {
    }
    function K(e) {
      var t = [];
      return e.forEach(function(a) {
        t.push(a);
      }), t.sort().join(", ");
    }
    function D(e, t, a, i) {
      return new qf(e, t, a, i);
    }
    function ue(e, t) {
      e.context === uf && (Tt(e.current, 2, t, e, null, null), Oc());
    }
    function Oe(e, t) {
      if (ru !== null) {
        var a = t.staleFamilies;
        t = t.updatedFamilies, Ho(), wf(
          e.current,
          t,
          a
        ), Oc();
      }
    }
    function ot(e) {
      ru = e;
    }
    function He(e) {
      return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11);
    }
    function Pe(e) {
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
    function Ct(e) {
      if (e.tag === 13) {
        var t = e.memoizedState;
        if (t === null && (e = e.alternate, e !== null && (t = e.memoizedState)), t !== null) return t.dehydrated;
      }
      return null;
    }
    function Ke(e) {
      if (Pe(e) !== e)
        throw Error("Unable to find node on an unmounted component.");
    }
    function At(e) {
      var t = e.alternate;
      if (!t) {
        if (t = Pe(e), t === null)
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
            if (f === a) return Ke(o), e;
            if (f === i) return Ke(o), t;
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
    function be(e) {
      var t = e.tag;
      if (t === 5 || t === 26 || t === 27 || t === 6) return e;
      for (e = e.child; e !== null; ) {
        if (t = be(e), t !== null) return t;
        e = e.sibling;
      }
      return null;
    }
    function pt(e) {
      return e === null || typeof e != "object" ? null : (e = Am && e[Am] || e["@@iterator"], typeof e == "function" ? e : null);
    }
    function je(e) {
      if (e == null) return null;
      if (typeof e == "function")
        return e.$$typeof === Ld ? null : e.displayName || e.name || null;
      if (typeof e == "string") return e;
      switch (e) {
        case Ve:
          return "Fragment";
        case Jo:
          return "Profiler";
        case Ko:
          return "StrictMode";
        case ko:
          return "Suspense";
        case xi:
          return "SuspenseList";
        case Rm:
          return "Activity";
      }
      if (typeof e == "object")
        switch (typeof e.tag == "number" && console.error(
          "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
        ), e.$$typeof) {
          case wc:
            return "Portal";
          case Pa:
            return (e.displayName || "Context") + ".Provider";
          case Gd:
            return (e._context.displayName || "Context") + ".Consumer";
          case Lu:
            var t = e.render;
            return e = e.displayName, e || (e = t.displayName || t.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
          case _r:
            return t = e.displayName || null, t !== null ? t : je(e.type) || "Memo";
          case xa:
            t = e._payload, e = e._init;
            try {
              return je(e(t));
            } catch {
            }
        }
      return null;
    }
    function St(e) {
      return typeof e.tag == "number" ? de(e) : typeof e.name == "string" ? e.name : null;
    }
    function de(e) {
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
          return je(t);
        case 8:
          return t === Ko ? "StrictMode" : "Mode";
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
            return de(e.return);
      }
      return null;
    }
    function Ot(e) {
      return { current: e };
    }
    function ve(e, t) {
      0 > en ? console.error("Unexpected pop.") : (t !== Cr[en] && console.error("Unexpected Fiber popped."), e.current = Ur[en], Ur[en] = null, Cr[en] = null, en--);
    }
    function ze(e, t, a) {
      en++, Ur[en] = e.current, Cr[en] = a, e.current = t;
    }
    function Dt(e) {
      return e === null && console.error(
        "Expected host context to exist. This error is likely caused by a bug in React. Please file an issue."
      ), e;
    }
    function Ht(e, t) {
      ze(uu, t, e), ze($o, e, e), ze(Vu, null, e);
      var a = t.nodeType;
      switch (a) {
        case 9:
        case 11:
          a = a === 9 ? "#document" : "#fragment", t = (t = t.documentElement) && (t = t.namespaceURI) ? bt(t) : $c;
          break;
        default:
          if (a = t.tagName, t = t.namespaceURI)
            t = bt(t), t = ma(
              t,
              a
            );
          else
            switch (a) {
              case "svg":
                t = _h;
                break;
              case "math":
                t = rg;
                break;
              default:
                t = $c;
            }
      }
      a = a.toLowerCase(), a = Yh(null, a), a = {
        context: t,
        ancestorInfo: a
      }, ve(Vu, e), ze(Vu, a, e);
    }
    function le(e) {
      ve(Vu, e), ve($o, e), ve(uu, e);
    }
    function R() {
      return Dt(Vu.current);
    }
    function X(e) {
      e.memoizedState !== null && ze(Wo, e, e);
      var t = Dt(Vu.current), a = e.type, i = ma(t.context, a);
      a = Yh(t.ancestorInfo, a), i = { context: i, ancestorInfo: a }, t !== i && (ze($o, e, e), ze(Vu, i, e));
    }
    function I(e) {
      $o.current === e && (ve(Vu, e), ve($o, e)), Wo.current === e && (ve(Wo, e), bp._currentValue = us);
    }
    function ge(e) {
      return typeof Symbol == "function" && Symbol.toStringTag && e[Symbol.toStringTag] || e.constructor.name || "Object";
    }
    function g(e) {
      try {
        return w(e), !1;
      } catch {
        return !0;
      }
    }
    function w(e) {
      return "" + e;
    }
    function J(e, t) {
      if (g(e))
        return console.error(
          "The provided `%s` attribute is an unsupported type %s. This value must be coerced to a string before using it here.",
          t,
          ge(e)
        ), w(e);
    }
    function P(e, t) {
      if (g(e))
        return console.error(
          "The provided `%s` CSS property is an unsupported type %s. This value must be coerced to a string before using it here.",
          t,
          ge(e)
        ), w(e);
    }
    function ce(e) {
      if (g(e))
        return console.error(
          "Form field values (value, checked, defaultValue, or defaultChecked props) must be strings, not %s. This value must be coerced to a string before using it here.",
          ge(e)
        ), w(e);
    }
    function De(e) {
      if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u") return !1;
      var t = __REACT_DEVTOOLS_GLOBAL_HOOK__;
      if (t.isDisabled) return !0;
      if (!t.supportsFiber)
        return console.error(
          "The installed version of React DevTools is too old and will not work with the current version of React. Please update React DevTools. https://react.dev/link/react-devtools"
        ), !0;
      try {
        Ni = t.inject(e), ql = t;
      } catch (a) {
        console.error("React instrumentation encountered an error: %s.", a);
      }
      return !!t.checkDCE;
    }
    function oe(e) {
      if (typeof Gg == "function" && En(e), ql && typeof ql.setStrictMode == "function")
        try {
          ql.setStrictMode(Ni, e);
        } catch (t) {
          ga || (ga = !0, console.error(
            "React instrumentation encountered an error: %s",
            t
          ));
        }
    }
    function cl(e) {
      fe = e;
    }
    function xe() {
      fe !== null && typeof fe.markCommitStopped == "function" && fe.markCommitStopped();
    }
    function Bt(e) {
      fe !== null && typeof fe.markComponentRenderStarted == "function" && fe.markComponentRenderStarted(e);
    }
    function ua() {
      fe !== null && typeof fe.markComponentRenderStopped == "function" && fe.markComponentRenderStopped();
    }
    function Mn(e) {
      fe !== null && typeof fe.markRenderStarted == "function" && fe.markRenderStarted(e);
    }
    function Ki() {
      fe !== null && typeof fe.markRenderStopped == "function" && fe.markRenderStopped();
    }
    function _n(e, t) {
      fe !== null && typeof fe.markStateUpdateScheduled == "function" && fe.markStateUpdateScheduled(e, t);
    }
    function eo(e) {
      return e >>>= 0, e === 0 ? 32 : 31 - (Qd(e) / Qu | 0) | 0;
    }
    function Sf(e) {
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
    function ll(e) {
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
    function vl(e, t, a) {
      var i = e.pendingLanes;
      if (i === 0) return 0;
      var o = 0, f = e.suspendedLanes, d = e.pingedLanes;
      e = e.warmLanes;
      var h = i & 134217727;
      return h !== 0 ? (i = h & ~f, i !== 0 ? o = ll(i) : (d &= h, d !== 0 ? o = ll(d) : a || (a = h & ~e, a !== 0 && (o = ll(a))))) : (h = i & ~f, h !== 0 ? o = ll(h) : d !== 0 ? o = ll(d) : a || (a = i & ~e, a !== 0 && (o = ll(a)))), o === 0 ? 0 : t !== 0 && t !== o && (t & f) === 0 && (f = o & -o, a = t & -t, f >= a || f === 32 && (a & 4194048) !== 0) ? t : o;
    }
    function Pu(e, t) {
      return (e.pendingLanes & ~(e.suspendedLanes & ~e.pingedLanes) & t) === 0;
    }
    function cs(e, t) {
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
    function Je() {
      var e = Zd;
      return Zd <<= 1, (Zd & 4194048) === 0 && (Zd = 256), e;
    }
    function Un() {
      var e = Kd;
      return Kd <<= 1, (Kd & 62914560) === 0 && (Kd = 4194304), e;
    }
    function to(e) {
      for (var t = [], a = 0; 31 > a; a++) t.push(e);
      return t;
    }
    function Su(e, t) {
      e.pendingLanes |= t, t !== 268435456 && (e.suspendedLanes = 0, e.pingedLanes = 0, e.warmLanes = 0);
    }
    function os(e, t, a, i, o, f) {
      var d = e.pendingLanes;
      e.pendingLanes = a, e.suspendedLanes = 0, e.pingedLanes = 0, e.warmLanes = 0, e.expiredLanes &= a, e.entangledLanes &= a, e.errorRecoveryDisabledLanes &= a, e.shellSuspendCounter = 0;
      var h = e.entanglements, v = e.expirationTimes, b = e.hiddenUpdates;
      for (a = d & ~a; 0 < a; ) {
        var q = 31 - Zl(a), L = 1 << q;
        h[q] = 0, v[q] = -1;
        var x = b[q];
        if (x !== null)
          for (b[q] = null, q = 0; q < x.length; q++) {
            var V = x[q];
            V !== null && (V.lane &= -536870913);
          }
        a &= ~L;
      }
      i !== 0 && Tf(e, i, 0), f !== 0 && o === 0 && e.tag !== 0 && (e.suspendedLanes |= f & ~(d & ~t));
    }
    function Tf(e, t, a) {
      e.pendingLanes |= t, e.suspendedLanes &= ~t;
      var i = 31 - Zl(t);
      e.entangledLanes |= t, e.entanglements[i] = e.entanglements[i] | 1073741824 | a & 4194090;
    }
    function ei(e, t) {
      var a = e.entangledLanes |= t;
      for (e = e.entanglements; a; ) {
        var i = 31 - Zl(a), o = 1 << i;
        o & t | e[i] & t && (e[i] |= t), a &= ~o;
      }
    }
    function Dl(e) {
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
    function ja(e, t, a) {
      if (It)
        for (e = e.pendingUpdatersLaneMap; 0 < a; ) {
          var i = 31 - Zl(a), o = 1 << i;
          e[i].add(t), a &= ~o;
        }
    }
    function zl(e, t) {
      if (It)
        for (var a = e.pendingUpdatersLaneMap, i = e.memoizedUpdaters; 0 < t; ) {
          var o = 31 - Zl(t);
          e = 1 << o, o = a[o], 0 < o.size && (o.forEach(function(f) {
            var d = f.alternate;
            d !== null && i.has(d) || i.add(f);
          }), o.clear()), t &= ~e;
        }
    }
    function lo(e) {
      return e &= -e, Bl < e ? Rn < e ? (e & 134217727) !== 0 ? Zu : Jd : Rn : Bl;
    }
    function Ef() {
      var e = Ue.p;
      return e !== 0 ? e : (e = window.event, e === void 0 ? Zu : Yd(e.type));
    }
    function ao(e, t) {
      var a = Ue.p;
      try {
        return Ue.p = e, t();
      } finally {
        Ue.p = a;
      }
    }
    function un(e) {
      delete e[Kl], delete e[ba], delete e[Om], delete e[Ov], delete e[Io];
    }
    function ia(e) {
      var t = e[Kl];
      if (t) return t;
      for (var a = e.parentNode; a; ) {
        if (t = a[qi] || a[Kl]) {
          if (a = t.alternate, t.child !== null || a !== null && a.child !== null)
            for (e = Vo(e); e !== null; ) {
              if (a = e[Kl])
                return a;
              e = Vo(e);
            }
          return t;
        }
        e = a, a = e.parentNode;
      }
      return null;
    }
    function Ml(e) {
      if (e = e[Kl] || e[qi]) {
        var t = e.tag;
        if (t === 5 || t === 6 || t === 13 || t === 26 || t === 27 || t === 3)
          return e;
      }
      return null;
    }
    function cn(e) {
      var t = e.tag;
      if (t === 5 || t === 26 || t === 27 || t === 6)
        return e.stateNode;
      throw Error("getNodeFromInstance: Invalid argument.");
    }
    function m(e) {
      var t = e[Po];
      return t || (t = e[Po] = { hoistableStyles: /* @__PURE__ */ new Map(), hoistableScripts: /* @__PURE__ */ new Map() }), t;
    }
    function z(e) {
      e[ef] = !0;
    }
    function te(e, t) {
      ne(e, t), ne(e + "Capture", t);
    }
    function ne(e, t) {
      tn[e] && console.error(
        "EventRegistry: More than one plugin attempted to publish the same registration name, `%s`.",
        e
      ), tn[e] = t;
      var a = e.toLowerCase();
      for (qc[a] = e, e === "onDoubleClick" && (qc.ondblclick = e), e = 0; e < t.length; e++)
        Dv.add(t[e]);
    }
    function pe(e, t) {
      zv[t.type] || t.onChange || t.onInput || t.readOnly || t.disabled || t.value == null || console.error(
        e === "select" ? "You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set `onChange`." : "You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set either `onChange` or `readOnly`."
      ), t.onChange || t.readOnly || t.disabled || t.checked == null || console.error(
        "You provided a `checked` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultChecked`. Otherwise, set either `onChange` or `readOnly`."
      );
    }
    function we(e) {
      return Xu.call(Wd, e) ? !0 : Xu.call($d, e) ? !1 : kd.test(e) ? Wd[e] = !0 : ($d[e] = !0, console.error("Invalid attribute name: `%s`", e), !1);
    }
    function Ge(e, t, a) {
      if (we(t)) {
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
    function ut(e, t, a) {
      if (we(t))
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
    function Ye(e, t, a) {
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
    function al(e, t, a, i) {
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
    function on() {
    }
    function fs() {
      if (Bi === 0) {
        Dm = console.log, zm = console.info, Mv = console.warn, Mm = console.error, tf = console.group, _v = console.groupCollapsed, Uv = console.groupEnd;
        var e = {
          configurable: !0,
          enumerable: !0,
          value: on,
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
      Bi++;
    }
    function Hh() {
      if (Bi--, Bi === 0) {
        var e = { configurable: !0, enumerable: !0, writable: !0 };
        Object.defineProperties(console, {
          log: ke({}, e, { value: Dm }),
          info: ke({}, e, { value: zm }),
          warn: ke({}, e, { value: Mv }),
          error: ke({}, e, { value: Mm }),
          group: ke({}, e, { value: tf }),
          groupCollapsed: ke({}, e, { value: _v }),
          groupEnd: ke({}, e, { value: Uv })
        });
      }
      0 > Bi && console.error(
        "disabledDepth fell below zero. This is a bug in React. Please file an issue."
      );
    }
    function ol(e) {
      if (_m === void 0)
        try {
          throw Error();
        } catch (a) {
          var t = a.stack.trim().match(/\n( *(at )?)/);
          _m = t && t[1] || "", Nr = -1 < a.stack.indexOf(`
    at`) ? " (<anonymous>)" : -1 < a.stack.indexOf("@") ? "@unknown:0:0" : "";
        }
      return `
` + _m + e + Nr;
    }
    function gl(e, t) {
      if (!e || lf) return "";
      var a = wr.get(e);
      if (a !== void 0) return a;
      lf = !0, a = Error.prepareStackTrace, Error.prepareStackTrace = void 0;
      var i = null;
      i = Y.H, Y.H = null, fs();
      try {
        var o = {
          DetermineComponentFrameRoot: function() {
            try {
              if (t) {
                var x = function() {
                  throw Error();
                };
                if (Object.defineProperty(x.prototype, "props", {
                  set: function() {
                    throw Error();
                  }
                }), typeof Reflect == "object" && Reflect.construct) {
                  try {
                    Reflect.construct(x, []);
                  } catch (ye) {
                    var V = ye;
                  }
                  Reflect.construct(e, [], x);
                } else {
                  try {
                    x.call();
                  } catch (ye) {
                    V = ye;
                  }
                  e.call(x.prototype);
                }
              } else {
                try {
                  throw Error();
                } catch (ye) {
                  V = ye;
                }
                (x = e()) && typeof x.catch == "function" && x.catch(function() {
                });
              }
            } catch (ye) {
              if (ye && V && typeof ye.stack == "string")
                return [ye.stack, V.stack];
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
`), q = v.split(`
`);
          for (d = f = 0; f < b.length && !b[f].includes(
            "DetermineComponentFrameRoot"
          ); )
            f++;
          for (; d < q.length && !q[d].includes(
            "DetermineComponentFrameRoot"
          ); )
            d++;
          if (f === b.length || d === q.length)
            for (f = b.length - 1, d = q.length - 1; 1 <= f && 0 <= d && b[f] !== q[d]; )
              d--;
          for (; 1 <= f && 0 <= d; f--, d--)
            if (b[f] !== q[d]) {
              if (f !== 1 || d !== 1)
                do
                  if (f--, d--, 0 > d || b[f] !== q[d]) {
                    var L = `
` + b[f].replace(
                      " at new ",
                      " at "
                    );
                    return e.displayName && L.includes("<anonymous>") && (L = L.replace("<anonymous>", e.displayName)), typeof e == "function" && wr.set(e, L), L;
                  }
                while (1 <= f && 0 <= d);
              break;
            }
        }
      } finally {
        lf = !1, Y.H = i, Hh(), Error.prepareStackTrace = a;
      }
      return b = (b = e ? e.displayName || e.name : "") ? ol(b) : "", typeof e == "function" && wr.set(e, b), b;
    }
    function $l(e) {
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
    function no(e) {
      switch (e.tag) {
        case 26:
        case 27:
        case 5:
          return ol(e.type);
        case 16:
          return ol("Lazy");
        case 13:
          return ol("Suspense");
        case 19:
          return ol("SuspenseList");
        case 0:
        case 15:
          return gl(e.type, !1);
        case 11:
          return gl(e.type.render, !1);
        case 1:
          return gl(e.type, !0);
        case 31:
          return ol("Activity");
        default:
          return "";
      }
    }
    function rs(e) {
      try {
        var t = "";
        do {
          t += no(e);
          var a = e._debugInfo;
          if (a)
            for (var i = a.length - 1; 0 <= i; i--) {
              var o = a[i];
              if (typeof o.name == "string") {
                var f = t, d = o.env, h = ol(
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
    function zp(e) {
      return (e = e ? e.displayName || e.name : "") ? ol(e) : "";
    }
    function ss() {
      if (Ha === null) return null;
      var e = Ha._debugOwner;
      return e != null ? St(e) : null;
    }
    function Mp() {
      if (Ha === null) return "";
      var e = Ha;
      try {
        var t = "";
        switch (e.tag === 6 && (e = e.return), e.tag) {
          case 26:
          case 27:
          case 5:
            t += ol(e.type);
            break;
          case 13:
            t += ol("Suspense");
            break;
          case 19:
            t += ol("SuspenseList");
            break;
          case 31:
            t += ol("Activity");
            break;
          case 30:
          case 0:
          case 15:
          case 1:
            e._debugOwner || t !== "" || (t += zp(
              e.type
            ));
            break;
          case 11:
            e._debugOwner || t !== "" || (t += zp(
              e.type.render
            ));
        }
        for (; e; )
          if (typeof e.tag == "number") {
            var a = e;
            e = a._debugOwner;
            var i = a._debugStack;
            e && i && (typeof i != "string" && (a._debugStack = i = $l(i)), i !== "" && (t += `
` + i));
          } else if (e.debugStack != null) {
            var o = e.debugStack;
            (e = e.owner) && o && (t += `
` + $l(o));
          } else break;
        var f = t;
      } catch (d) {
        f = `
Error generating stack: ` + d.message + `
` + d.stack;
      }
      return f;
    }
    function he(e, t, a, i, o, f, d) {
      var h = Ha;
      Rf(e);
      try {
        return e !== null && e._debugTask ? e._debugTask.run(
          t.bind(null, a, i, o, f, d)
        ) : t(a, i, o, f, d);
      } finally {
        Rf(h);
      }
      throw Error(
        "runWithFiberInDEV should never be called in production. This is a bug in React."
      );
    }
    function Rf(e) {
      Y.getCurrentStack = e === null ? null : Mp, Sa = !1, Ha = e;
    }
    function _l(e) {
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
    function Ji(e) {
      var t = e.type;
      return (e = e.nodeName) && e.toLowerCase() === "input" && (t === "checkbox" || t === "radio");
    }
    function Af(e) {
      var t = Ji(e) ? "checked" : "value", a = Object.getOwnPropertyDescriptor(
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
    function Tu(e) {
      e._valueTracker || (e._valueTracker = Af(e));
    }
    function fl(e) {
      if (!e) return !1;
      var t = e._valueTracker;
      if (!t) return !0;
      var a = t.getValue(), i = "";
      return e && (i = Ji(e) ? e.checked ? "true" : "false" : e.value), e = i, e !== a ? (t.setValue(e), !0) : !1;
    }
    function Of(e) {
      if (e = e || (typeof document < "u" ? document : void 0), typeof e > "u") return null;
      try {
        return e.activeElement || e.body;
      } catch {
        return e.body;
      }
    }
    function Aa(e) {
      return e.replace(
        Lg,
        function(t) {
          return "\\" + t.charCodeAt(0).toString(16) + " ";
        }
      );
    }
    function ti(e, t) {
      t.checked === void 0 || t.defaultChecked === void 0 || Cm || (console.error(
        "%s contains an input of type %s with both checked and defaultChecked props. Input elements must be either controlled or uncontrolled (specify either the checked prop, or the defaultChecked prop, but not both). Decide between using a controlled or uncontrolled input element and remove one of these props. More info: https://react.dev/link/controlled-components",
        ss() || "A component",
        t.type
      ), Cm = !0), t.value === void 0 || t.defaultValue === void 0 || Um || (console.error(
        "%s contains an input of type %s with both value and defaultValue props. Input elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled input element and remove one of these props. More info: https://react.dev/link/controlled-components",
        ss() || "A component",
        t.type
      ), Um = !0);
    }
    function li(e, t, a, i, o, f, d, h) {
      e.name = "", d != null && typeof d != "function" && typeof d != "symbol" && typeof d != "boolean" ? (J(d, "type"), e.type = d) : e.removeAttribute("type"), t != null ? d === "number" ? (t === 0 && e.value === "" || e.value != t) && (e.value = "" + _l(t)) : e.value !== "" + _l(t) && (e.value = "" + _l(t)) : d !== "submit" && d !== "reset" || e.removeAttribute("value"), t != null ? ds(e, d, _l(t)) : a != null ? ds(e, d, _l(a)) : i != null && e.removeAttribute("value"), o == null && f != null && (e.defaultChecked = !!f), o != null && (e.checked = o && typeof o != "function" && typeof o != "symbol"), h != null && typeof h != "function" && typeof h != "symbol" && typeof h != "boolean" ? (J(h, "name"), e.name = "" + _l(h)) : e.removeAttribute("name");
    }
    function _p(e, t, a, i, o, f, d, h) {
      if (f != null && typeof f != "function" && typeof f != "symbol" && typeof f != "boolean" && (J(f, "type"), e.type = f), t != null || a != null) {
        if (!(f !== "submit" && f !== "reset" || t != null))
          return;
        a = a != null ? "" + _l(a) : "", t = t != null ? "" + _l(t) : a, h || t === e.value || (e.value = t), e.defaultValue = t;
      }
      i = i ?? o, i = typeof i != "function" && typeof i != "symbol" && !!i, e.checked = h ? e.checked : !!i, e.defaultChecked = !!i, d != null && typeof d != "function" && typeof d != "symbol" && typeof d != "boolean" && (J(d, "name"), e.name = d);
    }
    function ds(e, t, a) {
      t === "number" && Of(e.ownerDocument) === e || e.defaultValue === "" + a || (e.defaultValue = "" + a);
    }
    function Nh(e, t) {
      t.value == null && (typeof t.children == "object" && t.children !== null ? zr.Children.forEach(t.children, function(a) {
        a == null || typeof a == "string" || typeof a == "number" || typeof a == "bigint" || Hm || (Hm = !0, console.error(
          "Cannot infer the option value of complex children. Pass a `value` prop or use a plain string as children to <option>."
        ));
      }) : t.dangerouslySetInnerHTML == null || Fd || (Fd = !0, console.error(
        "Pass a `value` prop if you set dangerouslyInnerHTML so React knows which value should be selected."
      ))), t.selected == null || xm || (console.error(
        "Use the `defaultValue` or `value` props on <select> instead of setting `selected` on <option>."
      ), xm = !0);
    }
    function Up() {
      var e = ss();
      return e ? `

Check the render method of \`` + e + "`." : "";
    }
    function Eu(e, t, a, i) {
      if (e = e.options, t) {
        t = {};
        for (var o = 0; o < a.length; o++)
          t["$" + a[o]] = !0;
        for (a = 0; a < e.length; a++)
          o = t.hasOwnProperty("$" + e[a].value), e[a].selected !== o && (e[a].selected = o), o && i && (e[a].defaultSelected = !0);
      } else {
        for (a = "" + _l(a), t = null, o = 0; o < e.length; o++) {
          if (e[o].value === a) {
            e[o].selected = !0, i && (e[o].defaultSelected = !0);
            return;
          }
          t !== null || e[o].disabled || (t = e[o]);
        }
        t !== null && (t.selected = !0);
      }
    }
    function Df(e, t) {
      for (e = 0; e < qr.length; e++) {
        var a = qr[e];
        if (t[a] != null) {
          var i = qe(t[a]);
          t.multiple && !i ? console.error(
            "The `%s` prop supplied to <select> must be an array if `multiple` is true.%s",
            a,
            Up()
          ) : !t.multiple && i && console.error(
            "The `%s` prop supplied to <select> must be a scalar value if `multiple` is false.%s",
            a,
            Up()
          );
        }
      }
      t.value === void 0 || t.defaultValue === void 0 || Nm || (console.error(
        "Select elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled select element and remove one of these props. More info: https://react.dev/link/controlled-components"
      ), Nm = !0);
    }
    function Cn(e, t) {
      t.value === void 0 || t.defaultValue === void 0 || Cv || (console.error(
        "%s contains a textarea with both value and defaultValue props. Textarea elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled textarea and remove one of these props. More info: https://react.dev/link/controlled-components",
        ss() || "A component"
      ), Cv = !0), t.children != null && t.value == null && console.error(
        "Use the `defaultValue` or `value` props instead of setting children on <textarea>."
      );
    }
    function hs(e, t, a) {
      if (t != null && (t = "" + _l(t), t !== e.value && (e.value = t), a == null)) {
        e.defaultValue !== t && (e.defaultValue = t);
        return;
      }
      e.defaultValue = a != null ? "" + _l(a) : "";
    }
    function wh(e, t, a, i) {
      if (t == null) {
        if (i != null) {
          if (a != null)
            throw Error(
              "If you supply `defaultValue` on a <textarea>, do not pass children."
            );
          if (qe(i)) {
            if (1 < i.length)
              throw Error("<textarea> can only have at most one child.");
            i = i[0];
          }
          a = i;
        }
        a == null && (a = ""), t = a;
      }
      a = _l(t), e.defaultValue = a, i = e.textContent, i === a && i !== "" && i !== null && (e.value = i);
    }
    function ki(e, t) {
      return e.serverProps === void 0 && e.serverTail.length === 0 && e.children.length === 1 && 3 < e.distanceFromLeaf && e.distanceFromLeaf > 15 - t ? ki(e.children[0], t) : e;
    }
    function Wl(e) {
      return "  " + "  ".repeat(e);
    }
    function ai(e) {
      return "+ " + "  ".repeat(e);
    }
    function $i(e) {
      return "- " + "  ".repeat(e);
    }
    function qh(e) {
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
    function Ll(e, t) {
      return xv.test(e) ? (e = JSON.stringify(e), e.length > t - 2 ? 8 > t ? '{"..."}' : "{" + e.slice(0, t - 7) + '..."}' : "{" + e + "}") : e.length > t ? 5 > t ? '{"..."}' : e.slice(0, t - 3) + "..." : e;
    }
    function zf(e, t, a) {
      var i = 120 - 2 * a;
      if (t === null)
        return ai(a) + Ll(e, i) + `
`;
      if (typeof t == "string") {
        for (var o = 0; o < t.length && o < e.length && t.charCodeAt(o) === e.charCodeAt(o); o++) ;
        return o > i - 8 && 10 < o && (e = "..." + e.slice(o - 8), t = "..." + t.slice(o - 8)), ai(a) + Ll(e, i) + `
` + $i(a) + Ll(t, i) + `
`;
      }
      return Wl(a) + Ll(e, i) + `
`;
    }
    function Bh(e) {
      return Object.prototype.toString.call(e).replace(/^\[object (.*)\]$/, function(t, a) {
        return a;
      });
    }
    function ni(e, t) {
      switch (typeof e) {
        case "string":
          return e = JSON.stringify(e), e.length > t ? 5 > t ? '"..."' : e.slice(0, t - 4) + '..."' : e;
        case "object":
          if (e === null) return "null";
          if (qe(e)) return "[...]";
          if (e.$$typeof === Ci)
            return (t = je(e.type)) ? "<" + t + ">" : "<...>";
          var a = Bh(e);
          if (a === "Object") {
            a = "", t -= 2;
            for (var i in e)
              if (e.hasOwnProperty(i)) {
                var o = JSON.stringify(i);
                if (o !== '"' + i + '"' && (i = o), t -= i.length - 2, o = ni(
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
    function Wi(e, t) {
      return typeof e != "string" || xv.test(e) ? "{" + ni(e, t - 2) + "}" : e.length > t - 2 ? 5 > t ? '"..."' : '"' + e.slice(0, t - 5) + '..."' : '"' + e + '"';
    }
    function uo(e, t, a) {
      var i = 120 - a.length - e.length, o = [], f;
      for (f in t)
        if (t.hasOwnProperty(f) && f !== "children") {
          var d = Wi(
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
    function Rg(e, t, a) {
      var i = "", o = ke({}, t), f;
      for (f in e)
        if (e.hasOwnProperty(f)) {
          delete o[f];
          var d = 120 - 2 * a - f.length - 2, h = ni(e[f], d);
          t.hasOwnProperty(f) ? (d = ni(t[f], d), i += ai(a) + f + ": " + h + `
`, i += $i(a) + f + ": " + d + `
`) : i += ai(a) + f + ": " + h + `
`;
        }
      for (var v in o)
        o.hasOwnProperty(v) && (e = ni(
          o[v],
          120 - 2 * a - v.length - 2
        ), i += $i(a) + v + ": " + e + `
`);
      return i;
    }
    function Ga(e, t, a, i) {
      var o = "", f = /* @__PURE__ */ new Map();
      for (b in a)
        a.hasOwnProperty(b) && f.set(
          b.toLowerCase(),
          b
        );
      if (f.size === 1 && f.has("children"))
        o += uo(
          e,
          t,
          Wl(i)
        );
      else {
        for (var d in t)
          if (t.hasOwnProperty(d) && d !== "children") {
            var h = 120 - 2 * (i + 1) - d.length - 1, v = f.get(d.toLowerCase());
            if (v !== void 0) {
              f.delete(d.toLowerCase());
              var b = t[d];
              v = a[v];
              var q = Wi(
                b,
                h
              );
              h = Wi(
                v,
                h
              ), typeof b == "object" && b !== null && typeof v == "object" && v !== null && Bh(b) === "Object" && Bh(v) === "Object" && (2 < Object.keys(b).length || 2 < Object.keys(v).length || -1 < q.indexOf("...") || -1 < h.indexOf("...")) ? o += Wl(i + 1) + d + `={{
` + Rg(
                b,
                v,
                i + 2
              ) + Wl(i + 1) + `}}
` : (o += ai(i + 1) + d + "=" + q + `
`, o += $i(i + 1) + d + "=" + h + `
`);
            } else
              o += Wl(i + 1) + d + "=" + Wi(t[d], h) + `
`;
          }
        f.forEach(function(L) {
          if (L !== "children") {
            var x = 120 - 2 * (i + 1) - L.length - 1;
            o += $i(i + 1) + L + "=" + Wi(a[L], x) + `
`;
          }
        }), o = o === "" ? Wl(i) + "<" + e + `>
` : Wl(i) + "<" + e + `
` + o + Wl(i) + `>
`;
      }
      return e = a.children, t = t.children, typeof e == "string" || typeof e == "number" || typeof e == "bigint" ? (f = "", (typeof t == "string" || typeof t == "number" || typeof t == "bigint") && (f = "" + t), o += zf(f, "" + e, i + 1)) : (typeof t == "string" || typeof t == "number" || typeof t == "bigint") && (o = e == null ? o + zf("" + t, null, i + 1) : o + zf("" + t, void 0, i + 1)), o;
    }
    function ys(e, t) {
      var a = qh(e);
      if (a === null) {
        for (a = "", e = e.child; e; )
          a += ys(e, t), e = e.sibling;
        return a;
      }
      return Wl(t) + "<" + a + `>
`;
    }
    function ms(e, t) {
      var a = ki(e, t);
      if (a !== e && (e.children.length !== 1 || e.children[0] !== a))
        return Wl(t) + `...
` + ms(a, t + 1);
      a = "";
      var i = e.fiber._debugInfo;
      if (i)
        for (var o = 0; o < i.length; o++) {
          var f = i[o].name;
          typeof f == "string" && (a += Wl(t) + "<" + f + `>
`, t++);
        }
      if (i = "", o = e.fiber.pendingProps, e.fiber.tag === 6)
        i = zf(o, e.serverProps, t), t++;
      else if (f = qh(e.fiber), f !== null)
        if (e.serverProps === void 0) {
          i = t;
          var d = 120 - 2 * i - f.length - 2, h = "";
          for (b in o)
            if (o.hasOwnProperty(b) && b !== "children") {
              var v = Wi(o[b], 15);
              if (d -= b.length + v.length + 2, 0 > d) {
                h += " ...";
                break;
              }
              h += " " + b + "=" + v;
            }
          i = Wl(i) + "<" + f + h + `>
`, t++;
        } else
          e.serverProps === null ? (i = uo(
            f,
            o,
            ai(t)
          ), t++) : typeof e.serverProps == "string" ? console.error(
            "Should not have matched a non HostText fiber to a Text node. This is a bug in React."
          ) : (i = Ga(
            f,
            o,
            e.serverProps,
            t
          ), t++);
      var b = "";
      for (o = e.fiber.child, f = 0; o && f < e.children.length; )
        d = e.children[f], d.fiber === o ? (b += ms(d, t), f++) : b += ys(o, t), o = o.sibling;
      for (o && 0 < e.children.length && (b += Wl(t) + `...
`), o = e.serverTail, e.serverProps === null && t--, e = 0; e < o.length; e++)
        f = o[e], b = typeof f == "string" ? b + ($i(t) + Ll(f, 120 - 2 * t) + `
`) : b + uo(
          f.type,
          f.props,
          $i(t)
        );
      return a + i + b;
    }
    function Mf(e) {
      try {
        return `

` + ms(e, 0);
      } catch {
        return "";
      }
    }
    function Fi(e, t, a) {
      for (var i = t, o = null, f = 0; i; )
        i === e && (f = 0), o = {
          fiber: i,
          children: o !== null ? [o] : [],
          serverProps: i === t ? a : i === e ? null : void 0,
          serverTail: [],
          distanceFromLeaf: f
        }, f++, i = i.return;
      return o !== null ? Mf(o).replaceAll(/^[+-]/gm, ">") : "";
    }
    function Yh(e, t) {
      var a = ke({}, e || Bm), i = { tag: t };
      return Id.indexOf(t) !== -1 && (a.aTagInScope = null, a.buttonTagInScope = null, a.nobrTagInScope = null), Pd.indexOf(t) !== -1 && (a.pTagInButtonScope = null), wm.indexOf(t) !== -1 && t !== "address" && t !== "div" && t !== "p" && (a.listItemTagAutoclosing = null, a.dlItemTagAutoclosing = null), a.current = i, t === "form" && (a.formTag = i), t === "a" && (a.aTagInScope = i), t === "button" && (a.buttonTagInScope = i), t === "nobr" && (a.nobrTagInScope = i), t === "p" && (a.pTagInButtonScope = i), t === "li" && (a.listItemTagAutoclosing = i), (t === "dd" || t === "dt") && (a.dlItemTagAutoclosing = i), t === "#document" || t === "html" ? a.containerTagInScope = null : a.containerTagInScope || (a.containerTagInScope = i), e !== null || t !== "#document" && t !== "html" && t !== "body" ? a.implicitRootScope === !0 && (a.implicitRootScope = !1) : a.implicitRootScope = !0, a;
    }
    function jh(e, t, a) {
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
          return qm.indexOf(t) === -1;
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
    function io(e, t) {
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
    function Cp(e, t) {
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
    function ps(e, t) {
      t = t || Bm;
      var a = t.current;
      if (t = (a = jh(
        e,
        a && a.tag,
        t.implicitRootScope
      ) ? null : a) ? null : io(e, t), t = a || t, !t) return !0;
      var i = t.tag;
      if (t = String(!!a) + "|" + e + "|" + i, af[t]) return !1;
      af[t] = !0;
      var o = (t = Ha) ? Cp(t.return, i) : null, f = t !== null && o !== null ? Fi(o, t, null) : "", d = "<" + e + ">";
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
      ), t && (e = t.return, o === null || e === null || o === e && e._debugOwner === t._debugOwner || he(o, function() {
        console.error(
          `<%s> cannot contain a nested %s.
See this log for the ancestor stack trace.`,
          i,
          d
        );
      })), !1;
    }
    function _f(e, t, a) {
      if (a || jh("#text", t, !1))
        return !0;
      if (a = "#text|" + t, af[a]) return !1;
      af[a] = !0;
      var i = (a = Ha) ? Cp(a, t) : null;
      return a = a !== null && i !== null ? Fi(
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
    function Ii(e, t) {
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
      return e.replace(Yi, function(t, a) {
        return a.toUpperCase();
      });
    }
    function xp(e, t, a) {
      var i = t.indexOf("--") === 0;
      i || (-1 < t.indexOf("-") ? Bc.hasOwnProperty(t) && Bc[t] || (Bc[t] = !0, console.error(
        "Unsupported style property %s. Did you mean %s?",
        t,
        Ag(t.replace(Yr, "ms-"))
      )) : Br.test(t) ? Bc.hasOwnProperty(t) && Bc[t] || (Bc[t] = !0, console.error(
        "Unsupported vendor-prefixed style property %s. Did you mean %s?",
        t,
        t.charAt(0).toUpperCase() + t.slice(1)
      )) : !Hv.test(a) || Yc.hasOwnProperty(a) && Yc[a] || (Yc[a] = !0, console.error(
        `Style property values shouldn't contain a semicolon. Try "%s: %s" instead.`,
        t,
        a.replace(Hv, "")
      )), typeof a == "number" && (isNaN(a) ? Nv || (Nv = !0, console.error(
        "`NaN` is an invalid value for the `%s` css style property.",
        t
      )) : isFinite(a) || Ym || (Ym = !0, console.error(
        "`Infinity` is an invalid value for the `%s` css style property.",
        t
      )))), a == null || typeof a == "boolean" || a === "" ? i ? e.setProperty(t, "") : t === "float" ? e.cssFloat = "" : e[t] = "" : i ? e.setProperty(t, a) : typeof a != "number" || a === 0 || jr.has(t) ? t === "float" ? e.cssFloat = a : (P(a, t), e[t] = ("" + a).trim()) : e[t] = a + "px";
    }
    function Uf(e, t, a) {
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
                for (var f = cu[o] || [o], d = 0; d < f.length; d++)
                  i[f[d]] = o;
          }
          for (var h in t)
            if (t.hasOwnProperty(h) && (!a || a[h] !== t[h]))
              for (o = cu[h] || [h], f = 0; f < o.length; f++)
                i[o[f]] = h;
          h = {};
          for (var v in t)
            for (o = cu[v] || [v], f = 0; f < o.length; f++)
              h[o[f]] = v;
          v = {};
          for (var b in i)
            if (o = i[b], (f = h[b]) && o !== f && (d = o + "," + f, !v[d])) {
              v[d] = !0, d = console;
              var q = t[o];
              d.error.call(
                d,
                "%s a style property during rerender (%s) when a conflicting property is set (%s) can lead to styling bugs. To avoid this, don't mix shorthand and non-shorthand properties for the same value; instead, replace the shorthand with separate values.",
                q == null || typeof q == "boolean" || q === "" ? "Removing" : "Updating",
                o,
                f
              );
            }
        }
        for (var L in a)
          !a.hasOwnProperty(L) || t != null && t.hasOwnProperty(L) || (L.indexOf("--") === 0 ? e.setProperty(L, "") : L === "float" ? e.cssFloat = "" : e[L] = "");
        for (var x in t)
          b = t[x], t.hasOwnProperty(x) && a[x] !== b && xp(e, x, b);
      } else
        for (i in t)
          t.hasOwnProperty(i) && xp(e, i, t[i]);
    }
    function Pi(e) {
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
    function vs(e) {
      return eh.get(e) || e;
    }
    function co(e, t) {
      if (Xu.call(ou, t) && ou[t])
        return !0;
      if (th.test(t)) {
        if (e = "aria-" + t.slice(4).toLowerCase(), e = jm.hasOwnProperty(e) ? e : null, e == null)
          return console.error(
            "Invalid ARIA attribute `%s`. ARIA attributes follow the pattern aria-* and must be lowercase.",
            t
          ), ou[t] = !0;
        if (t !== e)
          return console.error(
            "Invalid ARIA attribute `%s`. Did you mean `%s`?",
            t,
            e
          ), ou[t] = !0;
      }
      if (Gm.test(t)) {
        if (e = t.toLowerCase(), e = jm.hasOwnProperty(e) ? e : null, e == null) return ou[t] = !0, !1;
        t !== e && (console.error(
          "Unknown ARIA attribute `%s`. Did you mean `%s`?",
          t,
          e
        ), ou[t] = !0);
      }
      return !0;
    }
    function oo(e, t) {
      var a = [], i;
      for (i in t)
        co(e, i) || a.push(i);
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
    function Hp(e, t, a, i) {
      if (Xu.call(ta, t) && ta[t])
        return !0;
      var o = t.toLowerCase();
      if (o === "onfocusin" || o === "onfocusout")
        return console.error(
          "React uses onFocus and onBlur instead of onFocusIn and onFocusOut. All React events are normalized to bubble, so onFocusIn and onFocusOut are not needed/supported by React."
        ), ta[t] = !0;
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
          ), ta[t] = !0;
        if (Lr.test(t))
          return console.error(
            "Unknown event handler property `%s`. It will be ignored.",
            t
          ), ta[t] = !0;
      } else if (Lr.test(t))
        return l.test(t) && console.error(
          "Invalid event handler property `%s`. React events use the camelCase naming convention, for example `onClick`.",
          t
        ), ta[t] = !0;
      if (n.test(t) || u.test(t)) return !0;
      if (o === "innerhtml")
        return console.error(
          "Directly setting property `innerHTML` is not permitted. For more information, lookup documentation on `dangerouslySetInnerHTML`."
        ), ta[t] = !0;
      if (o === "aria")
        return console.error(
          "The `aria` attribute is reserved for future use in React. Pass individual `aria-` attributes instead."
        ), ta[t] = !0;
      if (o === "is" && a !== null && a !== void 0 && typeof a != "string")
        return console.error(
          "Received a `%s` for a string attribute `is`. If this is expected, cast the value to a string.",
          typeof a
        ), ta[t] = !0;
      if (typeof a == "number" && isNaN(a))
        return console.error(
          "Received NaN for the `%s` attribute. If this is expected, cast the value to a string.",
          t
        ), ta[t] = !0;
      if (jc.hasOwnProperty(o)) {
        if (o = jc[o], o !== t)
          return console.error(
            "Invalid DOM property `%s`. Did you mean `%s`?",
            t,
            o
          ), ta[t] = !0;
      } else if (t !== o)
        return console.error(
          "React does not recognize the `%s` prop on a DOM element. If you intentionally want it to appear in the DOM as a custom attribute, spell it as lowercase `%s` instead. If you accidentally passed it from a parent component, remove it from the DOM element.",
          t,
          o
        ), ta[t] = !0;
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
              ), ta[t] = !0);
          }
        case "function":
        case "symbol":
          return ta[t] = !0, !1;
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
            ), ta[t] = !0;
          }
      }
      return !0;
    }
    function Gh(e, t, a) {
      var i = [], o;
      for (o in t)
        Hp(e, o, t[o], a) || i.push(o);
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
    function fo(e) {
      return c.test("" + e) ? "javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')" : e;
    }
    function ec(e) {
      return e = e.target || e.srcElement || window, e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e;
    }
    function xn(e) {
      var t = Ml(e);
      if (t && (e = t.stateNode)) {
        var a = e[ba] || null;
        e: switch (e = t.stateNode, t.type) {
          case "input":
            if (li(
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
                  var o = i[ba] || null;
                  if (!o)
                    throw Error(
                      "ReactDOMInput: Mixing React and non-React radio inputs with the same `name` is not supported."
                    );
                  li(
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
                i = a[t], i.form === e.form && fl(i);
            }
            break e;
          case "textarea":
            hs(e, a.value, a.defaultValue);
            break e;
          case "select":
            t = a.value, t != null && Eu(e, !!a.multiple, t, !1);
        }
      }
    }
    function gs(e, t, a) {
      if (p) return e(t, a);
      p = !0;
      try {
        var i = e(t);
        return i;
      } finally {
        if (p = !1, (s !== null || y !== null) && (Oc(), s && (t = s, e = y, y = s = null, xn(t), e)))
          for (t = 0; t < e.length; t++) xn(e[t]);
      }
    }
    function Ru(e, t) {
      var a = e.stateNode;
      if (a === null) return null;
      var i = a[ba] || null;
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
    function Au() {
      if (B) return B;
      var e, t = N, a = t.length, i, o = "value" in W ? W.value : W.textContent, f = o.length;
      for (e = 0; e < a && t[e] === o[e]; e++) ;
      var d = a - e;
      for (i = 1; i <= d && t[a - i] === o[f - i]; i++) ;
      return B = o.slice(e, 1 < i ? 1 - i : void 0);
    }
    function ro(e) {
      var t = e.keyCode;
      return "charCode" in e ? (e = e.charCode, e === 0 && t === 13 && (e = 13)) : e = t, e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0;
    }
    function tc() {
      return !0;
    }
    function Lh() {
      return !1;
    }
    function Ul(e) {
      function t(a, i, o, f, d) {
        this._reactName = a, this._targetInst = o, this.type = i, this.nativeEvent = f, this.target = d, this.currentTarget = null;
        for (var h in e)
          e.hasOwnProperty(h) && (a = e[h], this[h] = a ? a(f) : f[h]);
        return this.isDefaultPrevented = (f.defaultPrevented != null ? f.defaultPrevented : f.returnValue === !1) ? tc : Lh, this.isPropagationStopped = Lh, this;
      }
      return ke(t.prototype, {
        preventDefault: function() {
          this.defaultPrevented = !0;
          var a = this.nativeEvent;
          a && (a.preventDefault ? a.preventDefault() : typeof a.returnValue != "unknown" && (a.returnValue = !1), this.isDefaultPrevented = tc);
        },
        stopPropagation: function() {
          var a = this.nativeEvent;
          a && (a.stopPropagation ? a.stopPropagation() : typeof a.cancelBubble != "unknown" && (a.cancelBubble = !0), this.isPropagationStopped = tc);
        },
        persist: function() {
        },
        isPersistent: tc
      }), t;
    }
    function bs(e) {
      var t = this.nativeEvent;
      return t.getModifierState ? t.getModifierState(e) : (e = pS[e]) ? !!t[e] : !1;
    }
    function Ss() {
      return bs;
    }
    function Fl(e, t) {
      switch (e) {
        case "keyup":
          return MS.indexOf(t.keyCode) !== -1;
        case "keydown":
          return t.keyCode !== K0;
        case "keypress":
        case "mousedown":
        case "focusout":
          return !0;
        default:
          return !1;
      }
    }
    function ui(e) {
      return e = e.detail, typeof e == "object" && "data" in e ? e.data : null;
    }
    function Ts(e, t) {
      switch (e) {
        case "compositionend":
          return ui(t);
        case "keypress":
          return t.which !== k0 ? null : (W0 = !0, $0);
        case "textInput":
          return e = t.data, e === $0 && W0 ? null : e;
        default:
          return null;
      }
    }
    function Cf(e, t) {
      if (lh)
        return e === "compositionend" || !Xg && Fl(e, t) ? (e = Au(), B = N = W = null, lh = !1, e) : null;
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
          return J0 && t.locale !== "ko" ? null : t.data;
        default:
          return null;
      }
    }
    function Np(e) {
      var t = e && e.nodeName && e.nodeName.toLowerCase();
      return t === "input" ? !!US[e.type] : t === "textarea";
    }
    function Vh(e) {
      if (!S) return !1;
      e = "on" + e;
      var t = e in document;
      return t || (t = document.createElement("div"), t.setAttribute(e, "return;"), t = typeof t[e] == "function"), t;
    }
    function Es(e, t, a, i) {
      s ? y ? y.push(i) : y = [i] : s = i, t = gr(t, "onChange"), 0 < t.length && (a = new Ee(
        "onChange",
        "change",
        null,
        a,
        i
      ), e.push({ event: a, listeners: t }));
    }
    function xf(e) {
      Pn(e, 0);
    }
    function lc(e) {
      var t = cn(e);
      if (fl(t)) return e;
    }
    function Xh(e, t) {
      if (e === "change") return t;
    }
    function wp() {
      Xm && (Xm.detachEvent("onpropertychange", qp), Qm = Xm = null);
    }
    function qp(e) {
      if (e.propertyName === "value" && lc(Qm)) {
        var t = [];
        Es(
          t,
          Qm,
          e,
          ec(e)
        ), gs(xf, t);
      }
    }
    function Og(e, t, a) {
      e === "focusin" ? (wp(), Xm = t, Qm = a, Xm.attachEvent("onpropertychange", qp)) : e === "focusout" && wp();
    }
    function Qh(e) {
      if (e === "selectionchange" || e === "keyup" || e === "keydown")
        return lc(Qm);
    }
    function Dg(e, t) {
      if (e === "click") return lc(t);
    }
    function zg(e, t) {
      if (e === "input" || e === "change")
        return lc(t);
    }
    function Mg(e, t) {
      return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t;
    }
    function Hf(e, t) {
      if (Na(e, t)) return !0;
      if (typeof e != "object" || e === null || typeof t != "object" || t === null)
        return !1;
      var a = Object.keys(e), i = Object.keys(t);
      if (a.length !== i.length) return !1;
      for (i = 0; i < a.length; i++) {
        var o = a[i];
        if (!Xu.call(t, o) || !Na(e[o], t[o]))
          return !1;
      }
      return !0;
    }
    function Bp(e) {
      for (; e && e.firstChild; ) e = e.firstChild;
      return e;
    }
    function Zh(e, t) {
      var a = Bp(e);
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
        a = Bp(a);
      }
    }
    function Yp(e, t) {
      return e && t ? e === t ? !0 : e && e.nodeType === 3 ? !1 : t && t.nodeType === 3 ? Yp(e, t.parentNode) : "contains" in e ? e.contains(t) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(t) & 16) : !1 : !1;
    }
    function jp(e) {
      e = e != null && e.ownerDocument != null && e.ownerDocument.defaultView != null ? e.ownerDocument.defaultView : window;
      for (var t = Of(e.document); t instanceof e.HTMLIFrameElement; ) {
        try {
          var a = typeof t.contentWindow.location.href == "string";
        } catch {
          a = !1;
        }
        if (a) e = t.contentWindow;
        else break;
        t = Of(e.document);
      }
      return t;
    }
    function Kh(e) {
      var t = e && e.nodeName && e.nodeName.toLowerCase();
      return t && (t === "input" && (e.type === "text" || e.type === "search" || e.type === "tel" || e.type === "url" || e.type === "password") || t === "textarea" || e.contentEditable === "true");
    }
    function Gp(e, t, a) {
      var i = a.window === a ? a.document : a.nodeType === 9 ? a : a.ownerDocument;
      Zg || ah == null || ah !== Of(i) || (i = ah, "selectionStart" in i && Kh(i) ? i = { start: i.selectionStart, end: i.selectionEnd } : (i = (i.ownerDocument && i.ownerDocument.defaultView || window).getSelection(), i = {
        anchorNode: i.anchorNode,
        anchorOffset: i.anchorOffset,
        focusNode: i.focusNode,
        focusOffset: i.focusOffset
      }), Zm && Hf(Zm, i) || (Zm = i, i = gr(Qg, "onSelect"), 0 < i.length && (t = new Ee(
        "onSelect",
        "select",
        null,
        t,
        a
      ), e.push({ event: t, listeners: i }), t.target = ah)));
    }
    function Ou(e, t) {
      var a = {};
      return a[e.toLowerCase()] = t.toLowerCase(), a["Webkit" + e] = "webkit" + t, a["Moz" + e] = "moz" + t, a;
    }
    function ac(e) {
      if (Kg[e]) return Kg[e];
      if (!nh[e]) return e;
      var t = nh[e], a;
      for (a in t)
        if (t.hasOwnProperty(a) && a in I0)
          return Kg[e] = t[a];
      return e;
    }
    function fn(e, t) {
      a1.set(e, t), te(t, [e]);
    }
    function Oa(e, t) {
      if (typeof e == "object" && e !== null) {
        var a = kg.get(e);
        return a !== void 0 ? a : (t = {
          value: e,
          source: t,
          stack: rs(t)
        }, kg.set(e, t), t);
      }
      return {
        value: e,
        source: t,
        stack: rs(t)
      };
    }
    function Nf() {
      for (var e = uh, t = $g = uh = 0; t < e; ) {
        var a = fu[t];
        fu[t++] = null;
        var i = fu[t];
        fu[t++] = null;
        var o = fu[t];
        fu[t++] = null;
        var f = fu[t];
        if (fu[t++] = null, i !== null && o !== null) {
          var d = i.pending;
          d === null ? o.next = o : (o.next = d.next, d.next = o), i.pending = o;
        }
        f !== 0 && Lp(a, o, f);
      }
    }
    function Rs(e, t, a, i) {
      fu[uh++] = e, fu[uh++] = t, fu[uh++] = a, fu[uh++] = i, $g |= i, e.lanes |= i, e = e.alternate, e !== null && (e.lanes |= i);
    }
    function Jh(e, t, a, i) {
      return Rs(e, t, a, i), As(e);
    }
    function ca(e, t) {
      return Rs(e, null, null, t), As(e);
    }
    function Lp(e, t, a) {
      e.lanes |= a;
      var i = e.alternate;
      i !== null && (i.lanes |= a);
      for (var o = !1, f = e.return; f !== null; )
        f.childLanes |= a, i = f.alternate, i !== null && (i.childLanes |= a), f.tag === 22 && (e = f.stateNode, e === null || e._visibility & wv || (o = !0)), e = f, f = f.return;
      return e.tag === 3 ? (f = e.stateNode, o && t !== null && (o = 31 - Zl(a), e = f.hiddenUpdates, i = e[o], i === null ? e[o] = [t] : i.push(t), t.lane = a | 536870912), f) : null;
    }
    function As(e) {
      if (hp > PS)
        throw es = hp = 0, yp = O0 = null, Error(
          "Maximum update depth exceeded. This can happen when a component repeatedly calls setState inside componentWillUpdate or componentDidUpdate. React limits the number of nested updates to prevent infinite loops."
        );
      es > eT && (es = 0, yp = null, console.error(
        "Maximum update depth exceeded. This can happen when a component calls setState inside useEffect, but useEffect either doesn't have a dependency array, or one of the dependencies changes on every render."
      )), e.alternate === null && (e.flags & 4098) !== 0 && Tn(e);
      for (var t = e, a = t.return; a !== null; )
        t.alternate === null && (t.flags & 4098) !== 0 && Tn(e), t = a, a = t.return;
      return t.tag === 3 ? t.stateNode : null;
    }
    function nc(e) {
      if (ru === null) return e;
      var t = ru(e);
      return t === void 0 ? e : t.current;
    }
    function kh(e) {
      if (ru === null) return e;
      var t = ru(e);
      return t === void 0 ? e != null && typeof e.render == "function" && (t = nc(e.render), e.render !== t) ? (t = { $$typeof: Lu, render: t }, e.displayName !== void 0 && (t.displayName = e.displayName), t) : e : t.current;
    }
    function Vp(e, t) {
      if (ru === null) return !1;
      var a = e.elementType;
      t = t.type;
      var i = !1, o = typeof t == "object" && t !== null ? t.$$typeof : null;
      switch (e.tag) {
        case 1:
          typeof t == "function" && (i = !0);
          break;
        case 0:
          (typeof t == "function" || o === xa) && (i = !0);
          break;
        case 11:
          (o === Lu || o === xa) && (i = !0);
          break;
        case 14:
        case 15:
          (o === _r || o === xa) && (i = !0);
          break;
        default:
          return !1;
      }
      return !!(i && (e = ru(a), e !== void 0 && e === ru(t)));
    }
    function Xp(e) {
      ru !== null && typeof WeakSet == "function" && (ih === null && (ih = /* @__PURE__ */ new WeakSet()), ih.add(e));
    }
    function wf(e, t, a) {
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
      if (ru === null)
        throw Error("Expected resolveFamily to be set during hot reload.");
      var b = !1;
      h = !1, v !== null && (v = ru(v), v !== void 0 && (a.has(v) ? h = !0 : t.has(v) && (d === 1 ? h = !0 : b = !0))), ih !== null && (ih.has(e) || i !== null && ih.has(i)) && (h = !0), h && (e._debugNeedsRemount = !0), (h || b) && (i = ca(e, 2), i !== null && Jt(i, e, 2)), o === null || h || wf(
        o,
        t,
        a
      ), f !== null && wf(
        f,
        t,
        a
      );
    }
    function qf(e, t, a, i) {
      this.tag = e, this.key = a, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.refCleanup = this.ref = null, this.pendingProps = t, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = i, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null, this.actualDuration = -0, this.actualStartTime = -1.1, this.treeBaseDuration = this.selfBaseDuration = -0, this._debugTask = this._debugStack = this._debugOwner = this._debugInfo = null, this._debugNeedsRemount = !1, this._debugHookTypes = null, u1 || typeof Object.preventExtensions != "function" || Object.preventExtensions(this);
    }
    function $h(e) {
      return e = e.prototype, !(!e || !e.isReactComponent);
    }
    function Hn(e, t) {
      var a = e.alternate;
      switch (a === null ? (a = D(
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
          a.type = nc(e.type);
          break;
        case 1:
          a.type = nc(e.type);
          break;
        case 11:
          a.type = kh(e.type);
      }
      return a;
    }
    function Wh(e, t) {
      e.flags &= 65011714;
      var a = e.alternate;
      return a === null ? (e.childLanes = 0, e.lanes = t, e.child = null, e.subtreeFlags = 0, e.memoizedProps = null, e.memoizedState = null, e.updateQueue = null, e.dependencies = null, e.stateNode = null, e.selfBaseDuration = 0, e.treeBaseDuration = 0) : (e.childLanes = a.childLanes, e.lanes = a.lanes, e.child = a.child, e.subtreeFlags = 0, e.deletions = null, e.memoizedProps = a.memoizedProps, e.memoizedState = a.memoizedState, e.updateQueue = a.updateQueue, e.type = a.type, t = a.dependencies, e.dependencies = t === null ? null : {
        lanes: t.lanes,
        firstContext: t.firstContext,
        _debugThenableState: t._debugThenableState
      }, e.selfBaseDuration = a.selfBaseDuration, e.treeBaseDuration = a.treeBaseDuration), e;
    }
    function Os(e, t, a, i, o, f) {
      var d = 0, h = e;
      if (typeof e == "function")
        $h(e) && (d = 1), h = nc(h);
      else if (typeof e == "string")
        d = R(), d = Qo(e, a, d) ? 26 : e === "html" || e === "head" || e === "body" ? 27 : 5;
      else
        e: switch (e) {
          case Rm:
            return t = D(31, a, t, o), t.elementType = Rm, t.lanes = f, t;
          case Ve:
            return ii(
              a.children,
              o,
              f,
              t
            );
          case Ko:
            d = 8, o |= Ta, o |= ku;
            break;
          case Jo:
            return e = a, i = o, typeof e.id != "string" && console.error(
              'Profiler must specify an "id" of type `string` as a prop. Received the type `%s` instead.',
              typeof e.id
            ), t = D(12, e, t, i | la), t.elementType = Jo, t.lanes = f, t.stateNode = { effectDuration: 0, passiveEffectDuration: 0 }, t;
          case ko:
            return t = D(13, a, t, o), t.elementType = ko, t.lanes = f, t;
          case xi:
            return t = D(19, a, t, o), t.elementType = xi, t.lanes = f, t;
          default:
            if (typeof e == "object" && e !== null)
              switch (e.$$typeof) {
                case Em:
                case Pa:
                  d = 10;
                  break e;
                case Gd:
                  d = 9;
                  break e;
                case Lu:
                  d = 11, h = kh(h);
                  break e;
                case _r:
                  d = 14;
                  break e;
                case xa:
                  d = 16, h = null;
                  break e;
              }
            h = "", (e === void 0 || typeof e == "object" && e !== null && Object.keys(e).length === 0) && (h += " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports."), e === null ? a = "null" : qe(e) ? a = "array" : e !== void 0 && e.$$typeof === Ci ? (a = "<" + (je(e.type) || "Unknown") + " />", h = " Did you accidentally export a JSX literal instead of a component?") : a = typeof e, (d = i ? St(i) : null) && (h += `

Check the render method of \`` + d + "`."), d = 29, a = Error(
              "Element type is invalid: expected a string (for built-in components) or a class/function (for composite components) but got: " + (a + "." + h)
            ), h = null;
        }
      return t = D(d, a, t, o), t.elementType = e, t.type = h, t.lanes = f, t._debugOwner = i, t;
    }
    function Bf(e, t, a) {
      return t = Os(
        e.type,
        e.key,
        e.props,
        e._owner,
        t,
        a
      ), t._debugOwner = e._owner, t._debugStack = e._debugStack, t._debugTask = e._debugTask, t;
    }
    function ii(e, t, a, i) {
      return e = D(7, e, i, t), e.lanes = a, e;
    }
    function ci(e, t, a) {
      return e = D(6, e, null, t), e.lanes = a, e;
    }
    function Fh(e, t, a) {
      return t = D(
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
    function uc(e, t) {
      rn(), ch[oh++] = Bv, ch[oh++] = qv, qv = e, Bv = t;
    }
    function Qp(e, t, a) {
      rn(), su[du++] = Lc, su[du++] = Vc, su[du++] = Vr, Vr = e;
      var i = Lc;
      e = Vc;
      var o = 32 - Zl(i) - 1;
      i &= ~(1 << o), a += 1;
      var f = 32 - Zl(t) + o;
      if (30 < f) {
        var d = o - o % 5;
        f = (i & (1 << d) - 1).toString(32), i >>= d, o -= d, Lc = 1 << 32 - Zl(t) + o | a << o | i, Vc = f + e;
      } else
        Lc = 1 << f | a << o | i, Vc = e;
    }
    function Ds(e) {
      rn(), e.return !== null && (uc(e, 1), Qp(e, 1, 0));
    }
    function zs(e) {
      for (; e === qv; )
        qv = ch[--oh], ch[oh] = null, Bv = ch[--oh], ch[oh] = null;
      for (; e === Vr; )
        Vr = su[--du], su[du] = null, Vc = su[--du], su[du] = null, Lc = su[--du], su[du] = null;
    }
    function rn() {
      mt || console.error(
        "Expected to be hydrating. This is a bug in React. Please file an issue."
      );
    }
    function sn(e, t) {
      if (e.return === null) {
        if (hu === null)
          hu = {
            fiber: e,
            children: [],
            serverProps: void 0,
            serverTail: [],
            distanceFromLeaf: t
          };
        else {
          if (hu.fiber !== e)
            throw Error(
              "Saw multiple hydration diff roots in a pass. This is a bug in React."
            );
          hu.distanceFromLeaf > t && (hu.distanceFromLeaf = t);
        }
        return hu;
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
    function Ih(e, t) {
      Xc || (e = sn(e, 0), e.serverProps = null, t !== null && (t = Ud(t), e.serverTail.push(t)));
    }
    function Nn(e) {
      var t = "", a = hu;
      throw a !== null && (hu = null, t = Mf(a)), so(
        Oa(
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
      ), Wg;
    }
    function Ph(e) {
      var t = e.stateNode, a = e.type, i = e.memoizedProps;
      switch (t[Kl] = e, t[ba] = i, eu(a, i), a) {
        case "dialog":
          et("cancel", t), et("close", t);
          break;
        case "iframe":
        case "object":
        case "embed":
          et("load", t);
          break;
        case "video":
        case "audio":
          for (a = 0; a < mp.length; a++)
            et(mp[a], t);
          break;
        case "source":
          et("error", t);
          break;
        case "img":
        case "image":
        case "link":
          et("error", t), et("load", t);
          break;
        case "details":
          et("toggle", t);
          break;
        case "input":
          pe("input", i), et("invalid", t), ti(t, i), _p(
            t,
            i.value,
            i.defaultValue,
            i.checked,
            i.defaultChecked,
            i.type,
            i.name,
            !0
          ), Tu(t);
          break;
        case "option":
          Nh(t, i);
          break;
        case "select":
          pe("select", i), et("invalid", t), Df(t, i);
          break;
        case "textarea":
          pe("textarea", i), et("invalid", t), Cn(t, i), wh(
            t,
            i.value,
            i.defaultValue,
            i.children
          ), Tu(t);
      }
      a = i.children, typeof a != "string" && typeof a != "number" && typeof a != "bigint" || t.textContent === "" + a || i.suppressHydrationWarning === !0 || tm(t.textContent, a) ? (i.popover != null && (et("beforetoggle", t), et("toggle", t)), i.onScroll != null && et("scroll", t), i.onScrollEnd != null && et("scrollend", t), i.onClick != null && (t.onclick = Bu), t = !0) : t = !1, t || Nn(e);
    }
    function ey(e) {
      for (wa = e.return; wa; )
        switch (wa.tag) {
          case 5:
          case 13:
            Gi = !1;
            return;
          case 27:
          case 3:
            Gi = !0;
            return;
          default:
            wa = wa.return;
        }
    }
    function ic(e) {
      if (e !== wa) return !1;
      if (!mt)
        return ey(e), mt = !0, !1;
      var t = e.tag, a;
      if ((a = t !== 3 && t !== 27) && ((a = t === 5) && (a = e.type, a = !(a !== "form" && a !== "button") || tu(e.type, e.memoizedProps)), a = !a), a && ul) {
        for (a = ul; a; ) {
          var i = sn(e, 0), o = Ud(a);
          i.serverTail.push(o), a = o.type === "Suspense" ? fm(a) : wl(a.nextSibling);
        }
        Nn(e);
      }
      if (ey(e), t === 13) {
        if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e)
          throw Error(
            "Expected to have a hydrated suspense instance. This error is likely caused by a bug in React. Please file an issue."
          );
        ul = fm(e);
      } else
        t === 27 ? (t = ul, lu(e.type) ? (e = B0, B0 = null, ul = e) : ul = t) : ul = wa ? wl(e.stateNode.nextSibling) : null;
      return !0;
    }
    function cc() {
      ul = wa = null, Xc = mt = !1;
    }
    function ty() {
      var e = Xr;
      return e !== null && (Ya === null ? Ya = e : Ya.push.apply(
        Ya,
        e
      ), Xr = null), e;
    }
    function so(e) {
      Xr === null ? Xr = [e] : Xr.push(e);
    }
    function ly() {
      var e = hu;
      if (e !== null) {
        hu = null;
        for (var t = Mf(e); 0 < e.children.length; )
          e = e.children[0];
        he(e.fiber, function() {
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
    function Ms() {
      fh = Yv = null, rh = !1;
    }
    function oi(e, t, a) {
      ze(Fg, t._currentValue, e), t._currentValue = a, ze(Ig, t._currentRenderer, e), t._currentRenderer !== void 0 && t._currentRenderer !== null && t._currentRenderer !== f1 && console.error(
        "Detected multiple renderers concurrently rendering the same context provider. This is currently unsupported."
      ), t._currentRenderer = f1;
    }
    function Du(e, t) {
      e._currentValue = Fg.current;
      var a = Ig.current;
      ve(Ig, t), e._currentRenderer = a, ve(Fg, t);
    }
    function ay(e, t, a) {
      for (; e !== null; ) {
        var i = e.alternate;
        if ((e.childLanes & t) !== t ? (e.childLanes |= t, i !== null && (i.childLanes |= t)) : i !== null && (i.childLanes & t) !== t && (i.childLanes |= t), e === a) break;
        e = e.return;
      }
      e !== a && console.error(
        "Expected to find the propagation root when scheduling context work. This error is likely caused by a bug in React. Please file an issue."
      );
    }
    function ny(e, t, a, i) {
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
                f.lanes |= a, h = f.alternate, h !== null && (h.lanes |= a), ay(
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
          d.lanes |= a, f = d.alternate, f !== null && (f.lanes |= a), ay(
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
    function Cl(e, t, a, i) {
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
            Na(o.pendingProps.value, d.value) || (e !== null ? e.push(h) : e = [h]);
          }
        } else if (o === Wo.current) {
          if (d = o.alternate, d === null)
            throw Error("Should have a current fiber. This is a bug in React.");
          d.memoizedState.memoizedState !== o.memoizedState.memoizedState && (e !== null ? e.push(bp) : e = [bp]);
        }
        o = o.return;
      }
      e !== null && ny(
        t,
        e,
        a,
        i
      ), t.flags |= 262144;
    }
    function fi(e) {
      for (e = e.firstContext; e !== null; ) {
        if (!Na(
          e.context._currentValue,
          e.memoizedValue
        ))
          return !0;
        e = e.next;
      }
      return !1;
    }
    function ri(e) {
      Yv = e, fh = null, e = e.dependencies, e !== null && (e.firstContext = null);
    }
    function Nt(e) {
      return rh && console.error(
        "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
      ), uy(Yv, e);
    }
    function Yf(e, t) {
      return Yv === null && ri(e), uy(e, t);
    }
    function uy(e, t) {
      var a = t._currentValue;
      if (t = { context: t, memoizedValue: a, next: null }, fh === null) {
        if (e === null)
          throw Error(
            "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
          );
        fh = t, e.dependencies = {
          lanes: 0,
          firstContext: t,
          _debugThenableState: null
        }, e.flags |= 524288;
      } else fh = fh.next = t;
      return a;
    }
    function jf() {
      return {
        controller: new YS(),
        data: /* @__PURE__ */ new Map(),
        refCount: 0
      };
    }
    function oc(e) {
      e.controller.signal.aborted && console.warn(
        "A cache instance was retained after it was already freed. This likely indicates a bug in React."
      ), e.refCount++;
    }
    function wn(e) {
      e.refCount--, 0 > e.refCount && console.warn(
        "A cache instance was released after it was already freed. This likely indicates a bug in React."
      ), e.refCount === 0 && jS(GS, function() {
        e.controller.abort();
      });
    }
    function dn() {
      var e = Qr;
      return Qr = 0, e;
    }
    function si(e) {
      var t = Qr;
      return Qr = e, t;
    }
    function fc(e) {
      var t = Qr;
      return Qr += e, t;
    }
    function _s(e) {
      ln = sh(), 0 > e.actualStartTime && (e.actualStartTime = ln);
    }
    function zu(e) {
      if (0 <= ln) {
        var t = sh() - ln;
        e.actualDuration += t, e.selfBaseDuration = t, ln = -1;
      }
    }
    function rc(e) {
      if (0 <= ln) {
        var t = sh() - ln;
        e.actualDuration += t, ln = -1;
      }
    }
    function La() {
      if (0 <= ln) {
        var e = sh() - ln;
        ln = -1, Qr += e;
      }
    }
    function hn() {
      ln = sh();
    }
    function qn(e) {
      for (var t = e.child; t; )
        e.actualDuration += t.actualDuration, t = t.sibling;
    }
    function Zp(e, t) {
      if (Km === null) {
        var a = Km = [];
        Pg = 0, Zr = Wy(), dh = {
          status: "pending",
          value: void 0,
          then: function(i) {
            a.push(i);
          }
        };
      }
      return Pg++, t.then(iy, iy), t;
    }
    function iy() {
      if (--Pg === 0 && Km !== null) {
        dh !== null && (dh.status = "fulfilled");
        var e = Km;
        Km = null, Zr = 0, dh = null;
        for (var t = 0; t < e.length; t++) (0, e[t])();
      }
    }
    function Kp(e, t) {
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
    function cy() {
      var e = Kr.current;
      return e !== null ? e : wt.pooledCache;
    }
    function Us(e, t) {
      t === null ? ze(Kr, Kr.current, e) : ze(Kr, t.pool, e);
    }
    function Jp() {
      var e = cy();
      return e === null ? null : { parent: Yl._currentValue, pool: e };
    }
    function oy() {
      return { didWarnAboutUncachedPromise: !1, thenables: [] };
    }
    function fy(e) {
      return e = e.status, e === "fulfilled" || e === "rejected";
    }
    function ho() {
    }
    function Va(e, t, a) {
      Y.actQueue !== null && (Y.didUsePromise = !0);
      var i = e.thenables;
      switch (a = i[a], a === void 0 ? i.push(t) : a !== t && (e.didWarnAboutUncachedPromise || (e.didWarnAboutUncachedPromise = !0, console.error(
        "A component was suspended by an uncached promise. Creating promises inside a Client Component or hook is not yet supported, except via a Suspense-compatible library or framework."
      )), t.then(ho, ho), t = a), t.status) {
        case "fulfilled":
          return t.value;
        case "rejected":
          throw e = t.reason, Da(e), e;
        default:
          if (typeof t.status == "string")
            t.then(ho, ho);
          else {
            if (e = wt, e !== null && 100 < e.shellSuspendCounter)
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
              throw e = t.reason, Da(e), e;
          }
          throw ep = t, Qv = !0, Pm;
      }
    }
    function ry() {
      if (ep === null)
        throw Error(
          "Expected a suspended thenable. This is a bug in React. Please file an issue."
        );
      var e = ep;
      return ep = null, Qv = !1, e;
    }
    function Da(e) {
      if (e === Pm || e === Xv)
        throw Error(
          "Hooks are not supported inside an async component. This error is often caused by accidentally adding `'use client'` to a module that was originally written for the server."
        );
    }
    function oa(e) {
      e.updateQueue = {
        baseState: e.memoizedState,
        firstBaseUpdate: null,
        lastBaseUpdate: null,
        shared: { pending: null, lanes: 0, hiddenCallbacks: null },
        callbacks: null
      };
    }
    function di(e, t) {
      e = e.updateQueue, t.updateQueue === e && (t.updateQueue = {
        baseState: e.baseState,
        firstBaseUpdate: e.firstBaseUpdate,
        lastBaseUpdate: e.lastBaseUpdate,
        shared: e.shared,
        callbacks: null
      });
    }
    function Bn(e) {
      return {
        lane: e,
        tag: y1,
        payload: null,
        callback: null,
        next: null
      };
    }
    function yn(e, t, a) {
      var i = e.updateQueue;
      if (i === null) return null;
      if (i = i.shared, l0 === i && !v1) {
        var o = de(e);
        console.error(
          `An update (setState, replaceState, or forceUpdate) was scheduled from inside an update function. Update functions should be pure, with zero side-effects. Consider using componentDidUpdate or a callback.

Please update the following component: %s`,
          o
        ), v1 = !0;
      }
      return (Et & Ba) !== An ? (o = i.pending, o === null ? t.next = t : (t.next = o.next, o.next = t), i.pending = t, t = As(e), Lp(e, null, a), t) : (Rs(e, i, t, a), As(e));
    }
    function hi(e, t, a) {
      if (t = t.updateQueue, t !== null && (t = t.shared, (a & 4194048) !== 0)) {
        var i = t.lanes;
        i &= e.pendingLanes, a |= i, t.lanes = a, ei(e, a);
      }
    }
    function yo(e, t) {
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
    function Yn() {
      if (a0) {
        var e = dh;
        if (e !== null) throw e;
      }
    }
    function mo(e, t, a, i) {
      a0 = !1;
      var o = e.updateQueue;
      cf = !1, l0 = o.shared;
      var f = o.firstBaseUpdate, d = o.lastBaseUpdate, h = o.shared.pending;
      if (h !== null) {
        o.shared.pending = null;
        var v = h, b = v.next;
        v.next = null, d === null ? f = b : d.next = b, d = v;
        var q = e.alternate;
        q !== null && (q = q.updateQueue, h = q.lastBaseUpdate, h !== d && (h === null ? q.firstBaseUpdate = b : h.next = b, q.lastBaseUpdate = v));
      }
      if (f !== null) {
        var L = o.baseState;
        d = 0, q = b = v = null, h = f;
        do {
          var x = h.lane & -536870913, V = x !== h.lane;
          if (V ? (nt & x) === x : (i & x) === x) {
            x !== 0 && x === Zr && (a0 = !0), q !== null && (q = q.next = {
              lane: 0,
              tag: h.tag,
              payload: h.payload,
              callback: null,
              next: null
            });
            e: {
              x = e;
              var ye = h, Ce = t, qt = a;
              switch (ye.tag) {
                case m1:
                  if (ye = ye.payload, typeof ye == "function") {
                    rh = !0;
                    var ct = ye.call(
                      qt,
                      L,
                      Ce
                    );
                    if (x.mode & Ta) {
                      oe(!0);
                      try {
                        ye.call(qt, L, Ce);
                      } finally {
                        oe(!1);
                      }
                    }
                    rh = !1, L = ct;
                    break e;
                  }
                  L = ye;
                  break e;
                case t0:
                  x.flags = x.flags & -65537 | 128;
                case y1:
                  if (ct = ye.payload, typeof ct == "function") {
                    if (rh = !0, ye = ct.call(
                      qt,
                      L,
                      Ce
                    ), x.mode & Ta) {
                      oe(!0);
                      try {
                        ct.call(qt, L, Ce);
                      } finally {
                        oe(!1);
                      }
                    }
                    rh = !1;
                  } else ye = ct;
                  if (ye == null) break e;
                  L = ke({}, L, ye);
                  break e;
                case p1:
                  cf = !0;
              }
            }
            x = h.callback, x !== null && (e.flags |= 64, V && (e.flags |= 8192), V = o.callbacks, V === null ? o.callbacks = [x] : V.push(x));
          } else
            V = {
              lane: x,
              tag: h.tag,
              payload: h.payload,
              callback: h.callback,
              next: null
            }, q === null ? (b = q = V, v = L) : q = q.next = V, d |= x;
          if (h = h.next, h === null) {
            if (h = o.shared.pending, h === null)
              break;
            V = h, h = V.next, V.next = null, o.lastBaseUpdate = V, o.shared.pending = null;
          }
        } while (!0);
        q === null && (v = L), o.baseState = v, o.firstBaseUpdate = b, o.lastBaseUpdate = q, f === null && (o.shared.lanes = 0), sf |= d, e.lanes = d, e.memoizedState = L;
      }
      l0 = null;
    }
    function Gf(e, t) {
      if (typeof e != "function")
        throw Error(
          "Invalid argument passed as callback. Expected a function. Instead received: " + e
        );
      e.call(t);
    }
    function po(e, t) {
      var a = e.shared.hiddenCallbacks;
      if (a !== null)
        for (e.shared.hiddenCallbacks = null, e = 0; e < a.length; e++)
          Gf(a[e], t);
    }
    function kp(e, t) {
      var a = e.callbacks;
      if (a !== null)
        for (e.callbacks = null, e = 0; e < a.length; e++)
          Gf(a[e], t);
    }
    function fa(e, t) {
      var a = Xi;
      ze(Zv, a, e), ze(hh, t, e), Xi = a | t.baseLanes;
    }
    function Lf(e) {
      ze(Zv, Xi, e), ze(
        hh,
        hh.current,
        e
      );
    }
    function mn(e) {
      Xi = Zv.current, ve(hh, e), ve(Zv, e);
    }
    function $e() {
      var e = G;
      pu === null ? pu = [e] : pu.push(e);
    }
    function ee() {
      var e = G;
      if (pu !== null && (Zc++, pu[Zc] !== e)) {
        var t = de(Be);
        if (!g1.has(t) && (g1.add(t), pu !== null)) {
          for (var a = "", i = 0; i <= Zc; i++) {
            var o = pu[i], f = i === Zc ? e : o;
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
    function Xa(e) {
      e == null || qe(e) || console.error(
        "%s received a final argument that is not an array (instead, received `%s`). When specified, the final argument must be an array.",
        G,
        typeof e
      );
    }
    function vo() {
      var e = de(Be);
      S1.has(e) || (S1.add(e), console.error(
        "ReactDOM.useFormState has been renamed to React.useActionState. Please update %s to use React.useActionState.",
        e
      ));
    }
    function Vt() {
      throw Error(
        `Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:
1. You might have mismatching versions of React and the renderer (such as React DOM)
2. You might be breaking the Rules of Hooks
3. You might have more than one copy of React in the same app
See https://react.dev/link/invalid-hook-call for tips about how to debug and fix this problem.`
      );
    }
    function yi(e, t) {
      if (lp) return !1;
      if (t === null)
        return console.error(
          "%s received a final argument during this render, but not during the previous render. Even though the final argument is optional, its type cannot change between renders.",
          G
        ), !1;
      e.length !== t.length && console.error(
        `The final argument passed to %s changed size between renders. The order and size of this array must remain constant.

Previous: %s
Incoming: %s`,
        G,
        "[" + t.join(", ") + "]",
        "[" + e.join(", ") + "]"
      );
      for (var a = 0; a < t.length && a < e.length; a++)
        if (!Na(e[a], t[a])) return !1;
      return !0;
    }
    function mi(e, t, a, i, o, f) {
      of = f, Be = t, pu = e !== null ? e._debugHookTypes : null, Zc = -1, lp = e !== null && e.type !== t.type, (Object.prototype.toString.call(a) === "[object AsyncFunction]" || Object.prototype.toString.call(a) === "[object AsyncGeneratorFunction]") && (f = de(Be), n0.has(f) || (n0.add(f), console.error(
        "%s is an async Client Component. Only Server Components can be async at the moment. This error is often caused by accidentally adding `'use client'` to a module that was originally written for the server.",
        f === null ? "An unknown Component" : "<" + f + ">"
      ))), t.memoizedState = null, t.updateQueue = null, t.lanes = 0, Y.H = e !== null && e.memoizedState !== null ? i0 : pu !== null ? T1 : u0, kr = f = (t.mode & Ta) !== Gt;
      var d = c0(a, i, o);
      if (kr = !1, mh && (d = go(
        t,
        a,
        i,
        o
      )), f) {
        oe(!0);
        try {
          d = go(
            t,
            a,
            i,
            o
          );
        } finally {
          oe(!1);
        }
      }
      return Vf(e, t), d;
    }
    function Vf(e, t) {
      t._debugHookTypes = pu, t.dependencies === null ? Qc !== null && (t.dependencies = {
        lanes: 0,
        firstContext: null,
        _debugThenableState: Qc
      }) : t.dependencies._debugThenableState = Qc, Y.H = kv;
      var a = xt !== null && xt.next !== null;
      if (of = 0, pu = G = Al = xt = Be = null, Zc = -1, e !== null && (e.flags & 65011712) !== (t.flags & 65011712) && console.error(
        "Internal React error: Expected static flag was missing. Please notify the React team."
      ), Kv = !1, tp = 0, Qc = null, a)
        throw Error(
          "Rendered fewer hooks than expected. This may be caused by an accidental early return statement."
        );
      e === null || Jl || (e = e.dependencies, e !== null && fi(e) && (Jl = !0)), Qv ? (Qv = !1, e = !0) : e = !1, e && (t = de(t) || "Unknown", b1.has(t) || n0.has(t) || (b1.add(t), console.error(
        "`use` was called from inside a try/catch block. This is not allowed and can lead to unexpected behavior. To handle errors triggered by `use`, wrap your component in a error boundary."
      )));
    }
    function go(e, t, a, i) {
      Be = e;
      var o = 0;
      do {
        if (mh && (Qc = null), tp = 0, mh = !1, o >= VS)
          throw Error(
            "Too many re-renders. React limits the number of renders to prevent an infinite loop."
          );
        if (o += 1, lp = !1, Al = xt = null, e.updateQueue != null) {
          var f = e.updateQueue;
          f.lastEffect = null, f.events = null, f.stores = null, f.memoCache != null && (f.memoCache.index = 0);
        }
        Zc = -1, Y.H = E1, f = c0(t, a, i);
      } while (mh);
      return f;
    }
    function Qa() {
      var e = Y.H, t = e.useState()[0];
      return t = typeof t.then == "function" ? sc(t) : t, e = e.useState()[0], (xt !== null ? xt.memoizedState : null) !== e && (Be.flags |= 1024), t;
    }
    function ra() {
      var e = Jv !== 0;
      return Jv = 0, e;
    }
    function Mu(e, t, a) {
      t.updateQueue = e.updateQueue, t.flags = (t.mode & ku) !== Gt ? t.flags & -402655237 : t.flags & -2053, e.lanes &= ~a;
    }
    function pn(e) {
      if (Kv) {
        for (e = e.memoizedState; e !== null; ) {
          var t = e.queue;
          t !== null && (t.pending = null), e = e.next;
        }
        Kv = !1;
      }
      of = 0, pu = Al = xt = Be = null, Zc = -1, G = null, mh = !1, tp = Jv = 0, Qc = null;
    }
    function Zt() {
      var e = {
        memoizedState: null,
        baseState: null,
        baseQueue: null,
        queue: null,
        next: null
      };
      return Al === null ? Be.memoizedState = Al = e : Al = Al.next = e, Al;
    }
    function it() {
      if (xt === null) {
        var e = Be.alternate;
        e = e !== null ? e.memoizedState : null;
      } else e = xt.next;
      var t = Al === null ? Be.memoizedState : Al.next;
      if (t !== null)
        Al = t, xt = e;
      else {
        if (e === null)
          throw Be.alternate === null ? Error(
            "Update hook called on initial render. This is likely a bug in React. Please file an issue."
          ) : Error("Rendered more hooks than during the previous render.");
        xt = e, e = {
          memoizedState: xt.memoizedState,
          baseState: xt.baseState,
          baseQueue: xt.baseQueue,
          queue: xt.queue,
          next: null
        }, Al === null ? Be.memoizedState = Al = e : Al = Al.next = e;
      }
      return Al;
    }
    function Cs() {
      return { lastEffect: null, events: null, stores: null, memoCache: null };
    }
    function sc(e) {
      var t = tp;
      return tp += 1, Qc === null && (Qc = oy()), e = Va(Qc, e, t), t = Be, (Al === null ? t.memoizedState : Al.next) === null && (t = t.alternate, Y.H = t !== null && t.memoizedState !== null ? i0 : u0), e;
    }
    function jn(e) {
      if (e !== null && typeof e == "object") {
        if (typeof e.then == "function") return sc(e);
        if (e.$$typeof === Pa) return Nt(e);
      }
      throw Error("An unsupported type was passed to use(): " + String(e));
    }
    function el(e) {
      var t = null, a = Be.updateQueue;
      if (a !== null && (t = a.memoCache), t == null) {
        var i = Be.alternate;
        i !== null && (i = i.updateQueue, i !== null && (i = i.memoCache, i != null && (t = {
          data: i.data.map(function(o) {
            return o.slice();
          }),
          index: 0
        })));
      }
      if (t == null && (t = { data: [], index: 0 }), a === null && (a = Cs(), Be.updateQueue = a), a.memoCache = t, a = t.data[t.index], a === void 0 || lp)
        for (a = t.data[t.index] = Array(e), i = 0; i < e; i++)
          a[i] = Rv;
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
      var i = Zt();
      if (a !== void 0) {
        var o = a(t);
        if (kr) {
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
      }, i.queue = e, e = e.dispatch = by.bind(
        null,
        Be,
        e
      ), [i.memoizedState, e];
    }
    function Za(e) {
      var t = it();
      return Ka(t, xt, e);
    }
    function Ka(e, t, a) {
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
        var h = d = null, v = null, b = t, q = !1;
        do {
          var L = b.lane & -536870913;
          if (L !== b.lane ? (nt & L) === L : (of & L) === L) {
            var x = b.revertLane;
            if (x === 0)
              v !== null && (v = v.next = {
                lane: 0,
                revertLane: 0,
                action: b.action,
                hasEagerState: b.hasEagerState,
                eagerState: b.eagerState,
                next: null
              }), L === Zr && (q = !0);
            else if ((of & x) === x) {
              b = b.next, x === Zr && (q = !0);
              continue;
            } else
              L = {
                lane: 0,
                revertLane: b.revertLane,
                action: b.action,
                hasEagerState: b.hasEagerState,
                eagerState: b.eagerState,
                next: null
              }, v === null ? (h = v = L, d = f) : v = v.next = L, Be.lanes |= x, sf |= x;
            L = b.action, kr && a(f, L), f = b.hasEagerState ? b.eagerState : a(f, L);
          } else
            x = {
              lane: L,
              revertLane: b.revertLane,
              action: b.action,
              hasEagerState: b.hasEagerState,
              eagerState: b.eagerState,
              next: null
            }, v === null ? (h = v = x, d = f) : v = v.next = x, Be.lanes |= L, sf |= L;
          b = b.next;
        } while (b !== null && b !== t);
        if (v === null ? d = f : v.next = h, !Na(f, e.memoizedState) && (Jl = !0, q && (a = dh, a !== null)))
          throw a;
        e.memoizedState = f, e.baseState = d, e.baseQueue = v, i.lastRenderedState = f;
      }
      return o === null && (i.lanes = 0), [e.memoizedState, i.dispatch];
    }
    function dc(e) {
      var t = it(), a = t.queue;
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
        Na(f, t.memoizedState) || (Jl = !0), t.memoizedState = f, t.baseQueue === null && (t.baseState = f), a.lastRenderedState = f;
      }
      return [f, i];
    }
    function _u(e, t, a) {
      var i = Be, o = Zt();
      if (mt) {
        if (a === void 0)
          throw Error(
            "Missing getServerSnapshot, which is required for server-rendered content. Will revert to client rendering."
          );
        var f = a();
        yh || f === a() || (console.error(
          "The result of getServerSnapshot should be cached to avoid an infinite loop"
        ), yh = !0);
      } else {
        if (f = t(), yh || (a = t(), Na(f, a) || (console.error(
          "The result of getSnapshot should be cached to avoid an infinite loop"
        ), yh = !0)), wt === null)
          throw Error(
            "Expected a work-in-progress root. This is a bug in React. Please file an issue."
          );
        (nt & 124) !== 0 || sy(i, t, f);
      }
      return o.memoizedState = f, a = { value: f, getSnapshot: t }, o.queue = a, Ns(
        So.bind(null, i, a, e),
        [e]
      ), i.flags |= 2048, Ln(
        mu | jl,
        vi(),
        bo.bind(
          null,
          i,
          a,
          f,
          t
        ),
        null
      ), f;
    }
    function Xf(e, t, a) {
      var i = Be, o = it(), f = mt;
      if (f) {
        if (a === void 0)
          throw Error(
            "Missing getServerSnapshot, which is required for server-rendered content. Will revert to client rendering."
          );
        a = a();
      } else if (a = t(), !yh) {
        var d = t();
        Na(a, d) || (console.error(
          "The result of getSnapshot should be cached to avoid an infinite loop"
        ), yh = !0);
      }
      (d = !Na(
        (xt || o).memoizedState,
        a
      )) && (o.memoizedState = a, Jl = !0), o = o.queue;
      var h = So.bind(null, i, o, e);
      if (sl(2048, jl, h, [e]), o.getSnapshot !== t || d || Al !== null && Al.memoizedState.tag & mu) {
        if (i.flags |= 2048, Ln(
          mu | jl,
          vi(),
          bo.bind(
            null,
            i,
            o,
            a,
            t
          ),
          null
        ), wt === null)
          throw Error(
            "Expected a work-in-progress root. This is a bug in React. Please file an issue."
          );
        f || (of & 124) !== 0 || sy(i, t, a);
      }
      return a;
    }
    function sy(e, t, a) {
      e.flags |= 16384, e = { getSnapshot: t, value: a }, t = Be.updateQueue, t === null ? (t = Cs(), Be.updateQueue = t, t.stores = [e]) : (a = t.stores, a === null ? t.stores = [e] : a.push(e));
    }
    function bo(e, t, a, i) {
      t.value = a, t.getSnapshot = i, dy(t) && To(e);
    }
    function So(e, t, a) {
      return a(function() {
        dy(t) && To(e);
      });
    }
    function dy(e) {
      var t = e.getSnapshot;
      e = e.value;
      try {
        var a = t();
        return !Na(e, a);
      } catch {
        return !0;
      }
    }
    function To(e) {
      var t = ca(e, 2);
      t !== null && Jt(t, e, 2);
    }
    function Qf(e) {
      var t = Zt();
      if (typeof e == "function") {
        var a = e;
        if (e = a(), kr) {
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
    function Uu(e) {
      e = Qf(e);
      var t = e.queue, a = Oo.bind(null, Be, t);
      return t.dispatch = a, [e.memoizedState, a];
    }
    function vn(e) {
      var t = Zt();
      t.memoizedState = t.baseState = e;
      var a = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: null,
        lastRenderedState: null
      };
      return t.queue = a, t = Xs.bind(
        null,
        Be,
        !0,
        a
      ), a.dispatch = t, [e, t];
    }
    function Cu(e, t) {
      var a = it();
      return Gn(a, xt, e, t);
    }
    function Gn(e, t, a, i) {
      return e.baseState = a, Ka(
        e,
        xt,
        typeof i == "function" ? i : dt
      );
    }
    function xs(e, t) {
      var a = it();
      return xt !== null ? Gn(a, xt, e, t) : (a.baseState = e, [e, a.queue.dispatch]);
    }
    function hy(e, t, a, i, o) {
      if (Ff(e))
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
        Y.T !== null ? a(!0) : f.isTransition = !1, i(f), a = t.pending, a === null ? (f.next = t.pending = f, Eo(t, f)) : (f.next = a.next, t.pending = a.next = f);
      }
    }
    function Eo(e, t) {
      var a = t.action, i = t.payload, o = e.state;
      if (t.isTransition) {
        var f = Y.T, d = {};
        Y.T = d, Y.T._updatedFibers = /* @__PURE__ */ new Set();
        try {
          var h = a(o, i), v = Y.S;
          v !== null && v(d, h), Zf(e, t, h);
        } catch (b) {
          bl(e, t, b);
        } finally {
          Y.T = f, f === null && d._updatedFibers && (e = d._updatedFibers.size, d._updatedFibers.clear(), 10 < e && console.warn(
            "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
          ));
        }
      } else
        try {
          d = a(o, i), Zf(e, t, d);
        } catch (b) {
          bl(e, t, b);
        }
    }
    function Zf(e, t, a) {
      a !== null && typeof a == "object" && typeof a.then == "function" ? (a.then(
        function(i) {
          pi(e, t, i);
        },
        function(i) {
          return bl(e, t, i);
        }
      ), t.isTransition || console.error(
        "An async function with useActionState was called outside of a transition. This is likely not what you intended (for example, isPending will not update correctly). Either call the returned function inside startTransition, or pass it to an `action` or `formAction` prop."
      )) : pi(e, t, a);
    }
    function pi(e, t, a) {
      t.status = "fulfilled", t.value = a, Kf(t), e.state = a, t = e.pending, t !== null && (a = t.next, a === t ? e.pending = null : (a = a.next, t.next = a, Eo(e, a)));
    }
    function bl(e, t, a) {
      var i = e.pending;
      if (e.pending = null, i !== null) {
        i = i.next;
        do
          t.status = "rejected", t.reason = a, Kf(t), t = t.next;
        while (t !== i);
      }
      e.action = null;
    }
    function Kf(e) {
      e = e.listeners;
      for (var t = 0; t < e.length; t++) (0, e[t])();
    }
    function yy(e, t) {
      return t;
    }
    function Ro(e, t) {
      if (mt) {
        var a = wt.formState;
        if (a !== null) {
          e: {
            var i = Be;
            if (mt) {
              if (ul) {
                t: {
                  for (var o = ul, f = Gi; o.nodeType !== 8; ) {
                    if (!f) {
                      o = null;
                      break t;
                    }
                    if (o = wl(
                      o.nextSibling
                    ), o === null) {
                      o = null;
                      break t;
                    }
                  }
                  f = o.data, o = f === H0 || f === Sb ? o : null;
                }
                if (o) {
                  ul = wl(
                    o.nextSibling
                  ), i = o.data === H0;
                  break e;
                }
              }
              Nn(i);
            }
            i = !1;
          }
          i && (t = a[0]);
        }
      }
      return a = Zt(), a.memoizedState = a.baseState = t, i = {
        pending: null,
        lanes: 0,
        dispatch: null,
        lastRenderedReducer: yy,
        lastRenderedState: t
      }, a.queue = i, a = Oo.bind(
        null,
        Be,
        i
      ), i.dispatch = a, i = Qf(!1), f = Xs.bind(
        null,
        Be,
        !1,
        i.queue
      ), i = Zt(), o = {
        state: t,
        dispatch: null,
        action: e,
        pending: null
      }, i.queue = o, a = hy.bind(
        null,
        Be,
        o,
        f,
        a
      ), o.dispatch = a, i.memoizedState = e, [t, a, !1];
    }
    function Hs(e) {
      var t = it();
      return $p(t, xt, e);
    }
    function $p(e, t, a) {
      if (t = Ka(
        e,
        t,
        yy
      )[0], e = Za(dt)[0], typeof t == "object" && t !== null && typeof t.then == "function")
        try {
          var i = sc(t);
        } catch (d) {
          throw d === Pm ? Xv : d;
        }
      else i = t;
      t = it();
      var o = t.queue, f = o.dispatch;
      return a !== t.memoizedState && (Be.flags |= 2048, Ln(
        mu | jl,
        vi(),
        rl.bind(null, o, a),
        null
      )), [i, f, e];
    }
    function rl(e, t) {
      e.action = t;
    }
    function Ao(e) {
      var t = it(), a = xt;
      if (a !== null)
        return $p(t, a, e);
      it(), t = t.memoizedState, a = it();
      var i = a.queue.dispatch;
      return a.memoizedState = e, [t, i, !1];
    }
    function Ln(e, t, a, i) {
      return e = {
        tag: e,
        create: a,
        deps: i,
        inst: t,
        next: null
      }, t = Be.updateQueue, t === null && (t = Cs(), Be.updateQueue = t), a = t.lastEffect, a === null ? t.lastEffect = e.next = e : (i = a.next, a.next = e, e.next = i, t.lastEffect = e), e;
    }
    function vi() {
      return { destroy: void 0, resource: void 0 };
    }
    function Jf(e) {
      var t = Zt();
      return e = { current: e }, t.memoizedState = e;
    }
    function Ja(e, t, a, i) {
      var o = Zt();
      i = i === void 0 ? null : i, Be.flags |= e, o.memoizedState = Ln(
        mu | t,
        vi(),
        a,
        i
      );
    }
    function sl(e, t, a, i) {
      var o = it();
      i = i === void 0 ? null : i;
      var f = o.memoizedState.inst;
      xt !== null && i !== null && yi(i, xt.memoizedState.deps) ? o.memoizedState = Ln(t, f, a, i) : (Be.flags |= e, o.memoizedState = Ln(
        mu | t,
        f,
        a,
        i
      ));
    }
    function Ns(e, t) {
      (Be.mode & ku) !== Gt && (Be.mode & n1) === Gt ? Ja(276826112, jl, e, t) : Ja(8390656, jl, e, t);
    }
    function ws(e, t) {
      var a = 4194308;
      return (Be.mode & ku) !== Gt && (a |= 134217728), Ja(a, aa, e, t);
    }
    function Wp(e, t) {
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
    function qs(e, t, a) {
      typeof t != "function" && console.error(
        "Expected useImperativeHandle() second argument to be a function that creates a handle. Instead received: %s.",
        t !== null ? typeof t : "null"
      ), a = a != null ? a.concat([e]) : null;
      var i = 4194308;
      (Be.mode & ku) !== Gt && (i |= 134217728), Ja(
        i,
        aa,
        Wp.bind(null, t, e),
        a
      );
    }
    function Vn(e, t, a) {
      typeof t != "function" && console.error(
        "Expected useImperativeHandle() second argument to be a function that creates a handle. Instead received: %s.",
        t !== null ? typeof t : "null"
      ), a = a != null ? a.concat([e]) : null, sl(
        4,
        aa,
        Wp.bind(null, t, e),
        a
      );
    }
    function kf(e, t) {
      return Zt().memoizedState = [
        e,
        t === void 0 ? null : t
      ], e;
    }
    function hc(e, t) {
      var a = it();
      t = t === void 0 ? null : t;
      var i = a.memoizedState;
      return t !== null && yi(t, i[1]) ? i[0] : (a.memoizedState = [e, t], e);
    }
    function Bs(e, t) {
      var a = Zt();
      t = t === void 0 ? null : t;
      var i = e();
      if (kr) {
        oe(!0);
        try {
          e();
        } finally {
          oe(!1);
        }
      }
      return a.memoizedState = [i, t], i;
    }
    function gi(e, t) {
      var a = it();
      t = t === void 0 ? null : t;
      var i = a.memoizedState;
      if (t !== null && yi(t, i[1]))
        return i[0];
      if (i = e(), kr) {
        oe(!0);
        try {
          e();
        } finally {
          oe(!1);
        }
      }
      return a.memoizedState = [i, t], i;
    }
    function Ys(e, t) {
      var a = Zt();
      return Gs(a, e, t);
    }
    function $f(e, t) {
      var a = it();
      return Wf(
        a,
        xt.memoizedState,
        e,
        t
      );
    }
    function js(e, t) {
      var a = it();
      return xt === null ? Gs(a, e, t) : Wf(
        a,
        xt.memoizedState,
        e,
        t
      );
    }
    function Gs(e, t, a) {
      return a === void 0 || (of & 1073741824) !== 0 ? e.memoizedState = t : (e.memoizedState = a, e = iv(), Be.lanes |= e, sf |= e, a);
    }
    function Wf(e, t, a, i) {
      return Na(a, t) ? a : hh.current !== null ? (e = Gs(e, a, i), Na(e, t) || (Jl = !0), e) : (of & 42) === 0 ? (Jl = !0, e.memoizedState = a) : (e = iv(), Be.lanes |= e, sf |= e, t);
    }
    function my(e, t, a, i, o) {
      var f = Ue.p;
      Ue.p = f !== 0 && f < Rn ? f : Rn;
      var d = Y.T, h = {};
      Y.T = h, Xs(e, !1, t, a), h._updatedFibers = /* @__PURE__ */ new Set();
      try {
        var v = o(), b = Y.S;
        if (b !== null && b(h, v), v !== null && typeof v == "object" && typeof v.then == "function") {
          var q = Kp(
            v,
            i
          );
          xu(
            e,
            t,
            q,
            ya(e)
          );
        } else
          xu(
            e,
            t,
            i,
            ya(e)
          );
      } catch (L) {
        xu(
          e,
          t,
          { then: function() {
          }, status: "rejected", reason: L },
          ya(e)
        );
      } finally {
        Ue.p = f, Y.T = d, d === null && h._updatedFibers && (e = h._updatedFibers.size, h._updatedFibers.clear(), 10 < e && console.warn(
          "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
        ));
      }
    }
    function yc(e, t, a, i) {
      if (e.tag !== 5)
        throw Error(
          "Expected the form instance to be a HostComponent. This is a bug in React."
        );
      var o = py(e).queue;
      my(
        e,
        o,
        t,
        us,
        a === null ? ie : function() {
          return vy(e), a(i);
        }
      );
    }
    function py(e) {
      var t = e.memoizedState;
      if (t !== null) return t;
      t = {
        memoizedState: us,
        baseState: us,
        baseQueue: null,
        queue: {
          pending: null,
          lanes: 0,
          dispatch: null,
          lastRenderedReducer: dt,
          lastRenderedState: us
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
    function vy(e) {
      Y.T === null && console.error(
        "requestFormReset was called outside a transition or action. To fix, move to an action, or wrap with startTransition."
      );
      var t = py(e).next.queue;
      xu(
        e,
        t,
        {},
        ya(e)
      );
    }
    function Xn() {
      var e = Qf(!1);
      return e = my.bind(
        null,
        Be,
        e.queue,
        !0,
        !1
      ), Zt().memoizedState = e, [!1, e];
    }
    function Ls() {
      var e = Za(dt)[0], t = it().memoizedState;
      return [
        typeof e == "boolean" ? e : sc(e),
        t
      ];
    }
    function Vs() {
      var e = dc(dt)[0], t = it().memoizedState;
      return [
        typeof e == "boolean" ? e : sc(e),
        t
      ];
    }
    function sa() {
      return Nt(bp);
    }
    function Qn() {
      var e = Zt(), t = wt.identifierPrefix;
      if (mt) {
        var a = Vc, i = Lc;
        a = (i & ~(1 << 32 - Zl(i) - 1)).toString(32) + a, t = "«" + t + "R" + a, a = Jv++, 0 < a && (t += "H" + a.toString(32)), t += "»";
      } else
        a = LS++, t = "«" + t + "r" + a.toString(32) + "»";
      return e.memoizedState = t;
    }
    function mc() {
      return Zt().memoizedState = gy.bind(
        null,
        Be
      );
    }
    function gy(e, t) {
      for (var a = e.return; a !== null; ) {
        switch (a.tag) {
          case 24:
          case 3:
            var i = ya(a);
            e = Bn(i);
            var o = yn(a, e, i);
            o !== null && (Jt(o, a, i), hi(o, a, i)), a = jf(), t != null && o !== null && console.error(
              "The seed argument is not enabled outside experimental channels."
            ), e.payload = { cache: a };
            return;
        }
        a = a.return;
      }
    }
    function by(e, t, a) {
      var i = arguments;
      typeof i[3] == "function" && console.error(
        "State updates from the useState() and useReducer() Hooks don't support the second callback argument. To execute a side effect after rendering, declare it in the component body with useEffect()."
      ), i = ya(e);
      var o = {
        lane: i,
        revertLane: 0,
        action: a,
        hasEagerState: !1,
        eagerState: null,
        next: null
      };
      Ff(e) ? pc(t, o) : (o = Jh(e, t, o, i), o !== null && (Jt(o, e, i), If(o, t, i))), _n(e, i);
    }
    function Oo(e, t, a) {
      var i = arguments;
      typeof i[3] == "function" && console.error(
        "State updates from the useState() and useReducer() Hooks don't support the second callback argument. To execute a side effect after rendering, declare it in the component body with useEffect()."
      ), i = ya(e), xu(e, t, a, i), _n(e, i);
    }
    function xu(e, t, a, i) {
      var o = {
        lane: i,
        revertLane: 0,
        action: a,
        hasEagerState: !1,
        eagerState: null,
        next: null
      };
      if (Ff(e)) pc(t, o);
      else {
        var f = e.alternate;
        if (e.lanes === 0 && (f === null || f.lanes === 0) && (f = t.lastRenderedReducer, f !== null)) {
          var d = Y.H;
          Y.H = Wu;
          try {
            var h = t.lastRenderedState, v = f(h, a);
            if (o.hasEagerState = !0, o.eagerState = v, Na(v, h))
              return Rs(e, t, o, 0), wt === null && Nf(), !1;
          } catch {
          } finally {
            Y.H = d;
          }
        }
        if (a = Jh(e, t, o, i), a !== null)
          return Jt(a, e, i), If(a, t, i), !0;
      }
      return !1;
    }
    function Xs(e, t, a, i) {
      if (Y.T === null && Zr === 0 && console.error(
        "An optimistic state update occurred outside a transition or action. To fix, move the update to an action, or wrap with startTransition."
      ), i = {
        lane: 2,
        revertLane: Wy(),
        action: i,
        hasEagerState: !1,
        eagerState: null,
        next: null
      }, Ff(e)) {
        if (t)
          throw Error("Cannot update optimistic state while rendering.");
        console.error("Cannot call startTransition while rendering.");
      } else
        t = Jh(
          e,
          a,
          i,
          2
        ), t !== null && Jt(t, e, 2);
      _n(e, 2);
    }
    function Ff(e) {
      var t = e.alternate;
      return e === Be || t !== null && t === Be;
    }
    function pc(e, t) {
      mh = Kv = !0;
      var a = e.pending;
      a === null ? t.next = t : (t.next = a.next, a.next = t), e.pending = t;
    }
    function If(e, t, a) {
      if ((a & 4194048) !== 0) {
        var i = t.lanes;
        i &= e.pendingLanes, a |= i, t.lanes = a, ei(e, a);
      }
    }
    function Sl(e) {
      var t = Fe;
      return e != null && (Fe = t === null ? e : t.concat(e)), t;
    }
    function Do(e, t, a) {
      for (var i = Object.keys(e.props), o = 0; o < i.length; o++) {
        var f = i[o];
        if (f !== "children" && f !== "key") {
          t === null && (t = Bf(e, a.mode, 0), t._debugInfo = Fe, t.return = a), he(
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
    function zo(e) {
      var t = ap;
      return ap += 1, ph === null && (ph = oy()), Va(ph, e, t);
    }
    function ka(e, t) {
      t = t.props.ref, e.ref = t !== void 0 ? t : null;
    }
    function Le(e, t) {
      throw t.$$typeof === Mr ? Error(
        `A React Element from an older version of React was rendered. This is not supported. It can happen if:
- Multiple copies of the "react" package is used.
- A library pre-bundled an old copy of "react" or "react/jsx-runtime".
- A compiler tries to "inline" JSX instead of using the runtime.`
      ) : (e = Object.prototype.toString.call(t), Error(
        "Objects are not valid as a React child (found: " + (e === "[object Object]" ? "object with keys {" + Object.keys(t).join(", ") + "}" : e) + "). If you meant to render a collection of children, use an array instead."
      ));
    }
    function vt(e, t) {
      var a = de(e) || "Component";
      B1[a] || (B1[a] = !0, t = t.displayName || t.name || "Component", e.tag === 3 ? console.error(
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
    function Xt(e, t) {
      var a = de(e) || "Component";
      Y1[a] || (Y1[a] = !0, t = String(t), e.tag === 3 ? console.error(
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
    function Pf(e) {
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
        return T = Hn(T, E), T.index = 0, T.sibling = null, T;
      }
      function f(T, E, A) {
        return T.index = A, e ? (A = T.alternate, A !== null ? (A = A.index, A < E ? (T.flags |= 67108866, E) : A) : (T.flags |= 67108866, E)) : (T.flags |= 1048576, E);
      }
      function d(T) {
        return e && T.alternate === null && (T.flags |= 67108866), T;
      }
      function h(T, E, A, Q) {
        return E === null || E.tag !== 6 ? (E = ci(
          A,
          T.mode,
          Q
        ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Fe, E) : (E = o(E, A), E.return = T, E._debugInfo = Fe, E);
      }
      function v(T, E, A, Q) {
        var ae = A.type;
        return ae === Ve ? (E = q(
          T,
          E,
          A.props.children,
          Q,
          A.key
        ), Do(A, E, T), E) : E !== null && (E.elementType === ae || Vp(E, A) || typeof ae == "object" && ae !== null && ae.$$typeof === xa && ff(ae) === E.type) ? (E = o(E, A.props), ka(E, A), E.return = T, E._debugOwner = A._owner, E._debugInfo = Fe, E) : (E = Bf(A, T.mode, Q), ka(E, A), E.return = T, E._debugInfo = Fe, E);
      }
      function b(T, E, A, Q) {
        return E === null || E.tag !== 4 || E.stateNode.containerInfo !== A.containerInfo || E.stateNode.implementation !== A.implementation ? (E = Fh(A, T.mode, Q), E.return = T, E._debugInfo = Fe, E) : (E = o(E, A.children || []), E.return = T, E._debugInfo = Fe, E);
      }
      function q(T, E, A, Q, ae) {
        return E === null || E.tag !== 7 ? (E = ii(
          A,
          T.mode,
          Q,
          ae
        ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Fe, E) : (E = o(E, A), E.return = T, E._debugInfo = Fe, E);
      }
      function L(T, E, A) {
        if (typeof E == "string" && E !== "" || typeof E == "number" || typeof E == "bigint")
          return E = ci(
            "" + E,
            T.mode,
            A
          ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Fe, E;
        if (typeof E == "object" && E !== null) {
          switch (E.$$typeof) {
            case Ci:
              return A = Bf(
                E,
                T.mode,
                A
              ), ka(A, E), A.return = T, T = Sl(E._debugInfo), A._debugInfo = Fe, Fe = T, A;
            case wc:
              return E = Fh(
                E,
                T.mode,
                A
              ), E.return = T, E._debugInfo = Fe, E;
            case xa:
              var Q = Sl(E._debugInfo);
              return E = ff(E), T = L(T, E, A), Fe = Q, T;
          }
          if (qe(E) || pt(E))
            return A = ii(
              E,
              T.mode,
              A,
              null
            ), A.return = T, A._debugOwner = T, A._debugTask = T._debugTask, T = Sl(E._debugInfo), A._debugInfo = Fe, Fe = T, A;
          if (typeof E.then == "function")
            return Q = Sl(E._debugInfo), T = L(
              T,
              zo(E),
              A
            ), Fe = Q, T;
          if (E.$$typeof === Pa)
            return L(
              T,
              Yf(T, E),
              A
            );
          Le(T, E);
        }
        return typeof E == "function" && vt(T, E), typeof E == "symbol" && Xt(T, E), null;
      }
      function x(T, E, A, Q) {
        var ae = E !== null ? E.key : null;
        if (typeof A == "string" && A !== "" || typeof A == "number" || typeof A == "bigint")
          return ae !== null ? null : h(T, E, "" + A, Q);
        if (typeof A == "object" && A !== null) {
          switch (A.$$typeof) {
            case Ci:
              return A.key === ae ? (ae = Sl(A._debugInfo), T = v(
                T,
                E,
                A,
                Q
              ), Fe = ae, T) : null;
            case wc:
              return A.key === ae ? b(T, E, A, Q) : null;
            case xa:
              return ae = Sl(A._debugInfo), A = ff(A), T = x(
                T,
                E,
                A,
                Q
              ), Fe = ae, T;
          }
          if (qe(A) || pt(A))
            return ae !== null ? null : (ae = Sl(A._debugInfo), T = q(
              T,
              E,
              A,
              Q,
              null
            ), Fe = ae, T);
          if (typeof A.then == "function")
            return ae = Sl(A._debugInfo), T = x(
              T,
              E,
              zo(A),
              Q
            ), Fe = ae, T;
          if (A.$$typeof === Pa)
            return x(
              T,
              E,
              Yf(T, A),
              Q
            );
          Le(T, A);
        }
        return typeof A == "function" && vt(T, A), typeof A == "symbol" && Xt(T, A), null;
      }
      function V(T, E, A, Q, ae) {
        if (typeof Q == "string" && Q !== "" || typeof Q == "number" || typeof Q == "bigint")
          return T = T.get(A) || null, h(E, T, "" + Q, ae);
        if (typeof Q == "object" && Q !== null) {
          switch (Q.$$typeof) {
            case Ci:
              return A = T.get(
                Q.key === null ? A : Q.key
              ) || null, T = Sl(Q._debugInfo), E = v(
                E,
                A,
                Q,
                ae
              ), Fe = T, E;
            case wc:
              return T = T.get(
                Q.key === null ? A : Q.key
              ) || null, b(E, T, Q, ae);
            case xa:
              var Xe = Sl(Q._debugInfo);
              return Q = ff(Q), E = V(
                T,
                E,
                A,
                Q,
                ae
              ), Fe = Xe, E;
          }
          if (qe(Q) || pt(Q))
            return A = T.get(A) || null, T = Sl(Q._debugInfo), E = q(
              E,
              A,
              Q,
              ae,
              null
            ), Fe = T, E;
          if (typeof Q.then == "function")
            return Xe = Sl(Q._debugInfo), E = V(
              T,
              E,
              A,
              zo(Q),
              ae
            ), Fe = Xe, E;
          if (Q.$$typeof === Pa)
            return V(
              T,
              E,
              A,
              Yf(E, Q),
              ae
            );
          Le(E, Q);
        }
        return typeof Q == "function" && vt(E, Q), typeof Q == "symbol" && Xt(E, Q), null;
      }
      function ye(T, E, A, Q) {
        if (typeof A != "object" || A === null) return Q;
        switch (A.$$typeof) {
          case Ci:
          case wc:
            st(T, E, A);
            var ae = A.key;
            if (typeof ae != "string") break;
            if (Q === null) {
              Q = /* @__PURE__ */ new Set(), Q.add(ae);
              break;
            }
            if (!Q.has(ae)) {
              Q.add(ae);
              break;
            }
            he(E, function() {
              console.error(
                "Encountered two children with the same key, `%s`. Keys should be unique so that components maintain their identity across updates. Non-unique keys may cause children to be duplicated and/or omitted — the behavior is unsupported and could change in a future version.",
                ae
              );
            });
            break;
          case xa:
            A = ff(A), ye(T, E, A, Q);
        }
        return Q;
      }
      function Ce(T, E, A, Q) {
        for (var ae = null, Xe = null, me = null, Qe = E, Ze = E = 0, Lt = null; Qe !== null && Ze < A.length; Ze++) {
          Qe.index > Ze ? (Lt = Qe, Qe = null) : Lt = Qe.sibling;
          var ml = x(
            T,
            Qe,
            A[Ze],
            Q
          );
          if (ml === null) {
            Qe === null && (Qe = Lt);
            break;
          }
          ae = ye(
            T,
            ml,
            A[Ze],
            ae
          ), e && Qe && ml.alternate === null && t(T, Qe), E = f(ml, E, Ze), me === null ? Xe = ml : me.sibling = ml, me = ml, Qe = Lt;
        }
        if (Ze === A.length)
          return a(T, Qe), mt && uc(T, Ze), Xe;
        if (Qe === null) {
          for (; Ze < A.length; Ze++)
            Qe = L(T, A[Ze], Q), Qe !== null && (ae = ye(
              T,
              Qe,
              A[Ze],
              ae
            ), E = f(
              Qe,
              E,
              Ze
            ), me === null ? Xe = Qe : me.sibling = Qe, me = Qe);
          return mt && uc(T, Ze), Xe;
        }
        for (Qe = i(Qe); Ze < A.length; Ze++)
          Lt = V(
            Qe,
            T,
            Ze,
            A[Ze],
            Q
          ), Lt !== null && (ae = ye(
            T,
            Lt,
            A[Ze],
            ae
          ), e && Lt.alternate !== null && Qe.delete(
            Lt.key === null ? Ze : Lt.key
          ), E = f(
            Lt,
            E,
            Ze
          ), me === null ? Xe = Lt : me.sibling = Lt, me = Lt);
        return e && Qe.forEach(function(Fc) {
          return t(T, Fc);
        }), mt && uc(T, Ze), Xe;
      }
      function qt(T, E, A, Q) {
        if (A == null)
          throw Error("An iterable object provided no iterator.");
        for (var ae = null, Xe = null, me = E, Qe = E = 0, Ze = null, Lt = null, ml = A.next(); me !== null && !ml.done; Qe++, ml = A.next()) {
          me.index > Qe ? (Ze = me, me = null) : Ze = me.sibling;
          var Fc = x(T, me, ml.value, Q);
          if (Fc === null) {
            me === null && (me = Ze);
            break;
          }
          Lt = ye(
            T,
            Fc,
            ml.value,
            Lt
          ), e && me && Fc.alternate === null && t(T, me), E = f(Fc, E, Qe), Xe === null ? ae = Fc : Xe.sibling = Fc, Xe = Fc, me = Ze;
        }
        if (ml.done)
          return a(T, me), mt && uc(T, Qe), ae;
        if (me === null) {
          for (; !ml.done; Qe++, ml = A.next())
            me = L(T, ml.value, Q), me !== null && (Lt = ye(
              T,
              me,
              ml.value,
              Lt
            ), E = f(
              me,
              E,
              Qe
            ), Xe === null ? ae = me : Xe.sibling = me, Xe = me);
          return mt && uc(T, Qe), ae;
        }
        for (me = i(me); !ml.done; Qe++, ml = A.next())
          Ze = V(
            me,
            T,
            Qe,
            ml.value,
            Q
          ), Ze !== null && (Lt = ye(
            T,
            Ze,
            ml.value,
            Lt
          ), e && Ze.alternate !== null && me.delete(
            Ze.key === null ? Qe : Ze.key
          ), E = f(
            Ze,
            E,
            Qe
          ), Xe === null ? ae = Ze : Xe.sibling = Ze, Xe = Ze);
        return e && me.forEach(function(yT) {
          return t(T, yT);
        }), mt && uc(T, Qe), ae;
      }
      function ct(T, E, A, Q) {
        if (typeof A == "object" && A !== null && A.type === Ve && A.key === null && (Do(A, null, T), A = A.props.children), typeof A == "object" && A !== null) {
          switch (A.$$typeof) {
            case Ci:
              var ae = Sl(A._debugInfo);
              e: {
                for (var Xe = A.key; E !== null; ) {
                  if (E.key === Xe) {
                    if (Xe = A.type, Xe === Ve) {
                      if (E.tag === 7) {
                        a(
                          T,
                          E.sibling
                        ), Q = o(
                          E,
                          A.props.children
                        ), Q.return = T, Q._debugOwner = A._owner, Q._debugInfo = Fe, Do(A, Q, T), T = Q;
                        break e;
                      }
                    } else if (E.elementType === Xe || Vp(
                      E,
                      A
                    ) || typeof Xe == "object" && Xe !== null && Xe.$$typeof === xa && ff(Xe) === E.type) {
                      a(
                        T,
                        E.sibling
                      ), Q = o(E, A.props), ka(Q, A), Q.return = T, Q._debugOwner = A._owner, Q._debugInfo = Fe, T = Q;
                      break e;
                    }
                    a(T, E);
                    break;
                  } else t(T, E);
                  E = E.sibling;
                }
                A.type === Ve ? (Q = ii(
                  A.props.children,
                  T.mode,
                  Q,
                  A.key
                ), Q.return = T, Q._debugOwner = T, Q._debugTask = T._debugTask, Q._debugInfo = Fe, Do(A, Q, T), T = Q) : (Q = Bf(
                  A,
                  T.mode,
                  Q
                ), ka(Q, A), Q.return = T, Q._debugInfo = Fe, T = Q);
              }
              return T = d(T), Fe = ae, T;
            case wc:
              e: {
                for (ae = A, A = ae.key; E !== null; ) {
                  if (E.key === A)
                    if (E.tag === 4 && E.stateNode.containerInfo === ae.containerInfo && E.stateNode.implementation === ae.implementation) {
                      a(
                        T,
                        E.sibling
                      ), Q = o(
                        E,
                        ae.children || []
                      ), Q.return = T, T = Q;
                      break e;
                    } else {
                      a(T, E);
                      break;
                    }
                  else t(T, E);
                  E = E.sibling;
                }
                Q = Fh(
                  ae,
                  T.mode,
                  Q
                ), Q.return = T, T = Q;
              }
              return d(T);
            case xa:
              return ae = Sl(A._debugInfo), A = ff(A), T = ct(
                T,
                E,
                A,
                Q
              ), Fe = ae, T;
          }
          if (qe(A))
            return ae = Sl(A._debugInfo), T = Ce(
              T,
              E,
              A,
              Q
            ), Fe = ae, T;
          if (pt(A)) {
            if (ae = Sl(A._debugInfo), Xe = pt(A), typeof Xe != "function")
              throw Error(
                "An object is not an iterable. This error is likely caused by a bug in React. Please file an issue."
              );
            var me = Xe.call(A);
            return me === A ? (T.tag !== 0 || Object.prototype.toString.call(T.type) !== "[object GeneratorFunction]" || Object.prototype.toString.call(me) !== "[object Generator]") && (w1 || console.error(
              "Using Iterators as children is unsupported and will likely yield unexpected results because enumerating a generator mutates it. You may convert it to an array with `Array.from()` or the `[...spread]` operator before rendering. You can also use an Iterable that can iterate multiple times over the same items."
            ), w1 = !0) : A.entries !== Xe || f0 || (console.error(
              "Using Maps as children is not supported. Use an array of keyed ReactElements instead."
            ), f0 = !0), T = qt(
              T,
              E,
              me,
              Q
            ), Fe = ae, T;
          }
          if (typeof A.then == "function")
            return ae = Sl(A._debugInfo), T = ct(
              T,
              E,
              zo(A),
              Q
            ), Fe = ae, T;
          if (A.$$typeof === Pa)
            return ct(
              T,
              E,
              Yf(T, A),
              Q
            );
          Le(T, A);
        }
        return typeof A == "string" && A !== "" || typeof A == "number" || typeof A == "bigint" ? (ae = "" + A, E !== null && E.tag === 6 ? (a(
          T,
          E.sibling
        ), Q = o(E, ae), Q.return = T, T = Q) : (a(T, E), Q = ci(
          ae,
          T.mode,
          Q
        ), Q.return = T, Q._debugOwner = T, Q._debugTask = T._debugTask, Q._debugInfo = Fe, T = Q), d(T)) : (typeof A == "function" && vt(T, A), typeof A == "symbol" && Xt(T, A), a(T, E));
      }
      return function(T, E, A, Q) {
        var ae = Fe;
        Fe = null;
        try {
          ap = 0;
          var Xe = ct(
            T,
            E,
            A,
            Q
          );
          return ph = null, Xe;
        } catch (Lt) {
          if (Lt === Pm || Lt === Xv) throw Lt;
          var me = D(29, Lt, null, T.mode);
          me.lanes = Q, me.return = T;
          var Qe = me._debugInfo = Fe;
          if (me._debugOwner = T._debugOwner, me._debugTask = T._debugTask, Qe != null) {
            for (var Ze = Qe.length - 1; 0 <= Ze; Ze--)
              if (typeof Qe[Ze].stack == "string") {
                me._debugOwner = Qe[Ze], me._debugTask = Qe[Ze].debugTask;
                break;
              }
          }
          return me;
        } finally {
          Fe = ae;
        }
      };
    }
    function za(e) {
      var t = e.alternate;
      ze(
        Gl,
        Gl.current & gh,
        e
      ), ze(vu, e, e), Vi === null && (t === null || hh.current !== null || t.memoizedState !== null) && (Vi = e);
    }
    function bi(e) {
      if (e.tag === 22) {
        if (ze(Gl, Gl.current, e), ze(vu, e, e), Vi === null) {
          var t = e.alternate;
          t !== null && t.memoizedState !== null && (Vi = e);
        }
      } else gn(e);
    }
    function gn(e) {
      ze(Gl, Gl.current, e), ze(
        vu,
        vu.current,
        e
      );
    }
    function Ma(e) {
      ve(vu, e), Vi === e && (Vi = null), ve(Gl, e);
    }
    function Hu(e) {
      for (var t = e; t !== null; ) {
        if (t.tag === 13) {
          var a = t.memoizedState;
          if (a !== null && (a = a.dehydrated, a === null || a.data === kc || au(a)))
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
    function Sy(e) {
      if (e !== null && typeof e != "function") {
        var t = String(e);
        W1.has(t) || (W1.add(t), console.error(
          "Expected the last optional `callback` argument to be a function. Instead received: %s.",
          e
        ));
      }
    }
    function Qt(e, t, a, i) {
      var o = e.memoizedState, f = a(i, o);
      if (e.mode & Ta) {
        oe(!0);
        try {
          f = a(i, o);
        } finally {
          oe(!1);
        }
      }
      f === void 0 && (t = je(t) || "Component", K1.has(t) || (K1.add(t), console.error(
        "%s.getDerivedStateFromProps(): A valid state object (or null) must be returned. You have returned undefined.",
        t
      ))), o = f == null ? o : ke({}, o, f), e.memoizedState = o, e.lanes === 0 && (e.updateQueue.baseState = o);
    }
    function Qs(e, t, a, i, o, f, d) {
      var h = e.stateNode;
      if (typeof h.shouldComponentUpdate == "function") {
        if (a = h.shouldComponentUpdate(
          i,
          f,
          d
        ), e.mode & Ta) {
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
          je(t) || "Component"
        ), a;
      }
      return t.prototype && t.prototype.isPureReactComponent ? !Hf(a, i) || !Hf(o, f) : !0;
    }
    function Zs(e, t, a, i) {
      var o = t.state;
      typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps(a, i), typeof t.UNSAFE_componentWillReceiveProps == "function" && t.UNSAFE_componentWillReceiveProps(a, i), t.state !== o && (e = de(e) || "Component", L1.has(e) || (L1.add(e), console.error(
        "%s.componentWillReceiveProps(): Assigning directly to this.state is deprecated (except inside a component's constructor). Use setState instead.",
        e
      )), r0.enqueueReplaceState(
        t,
        t.state,
        null
      ));
    }
    function Si(e, t) {
      var a = t;
      if ("ref" in t) {
        a = {};
        for (var i in t)
          i !== "ref" && (a[i] = t[i]);
      }
      if (e = e.defaultProps) {
        a === t && (a = ke({}, a));
        for (var o in e)
          a[o] === void 0 && (a[o] = e[o]);
      }
      return a;
    }
    function Ty(e) {
      s0(e), console.warn(
        `%s

%s
`,
        bh ? "An error occurred in the <" + bh + "> component." : "An error occurred in one of your React components.",
        `Consider adding an error boundary to your tree to customize error handling behavior.
Visit https://react.dev/link/error-boundaries to learn more about error boundaries.`
      );
    }
    function Fp(e) {
      var t = bh ? "The above error occurred in the <" + bh + "> component." : "The above error occurred in one of your React components.", a = "React will try to recreate this component tree from scratch using the error boundary you provided, " + ((d0 || "Anonymous") + ".");
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
          Mb + e[0],
          _b,
          hg + i + hg,
          Ub
        ) : e.splice(
          0,
          0,
          Mb,
          _b,
          hg + i + hg,
          Ub
        ), e.unshift(console), i = dT.apply(console.error, e), i();
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
    function Ks(e) {
      s0(e);
    }
    function Mo(e, t) {
      try {
        bh = t.source ? de(t.source) : null, d0 = null;
        var a = t.value;
        if (Y.actQueue !== null)
          Y.thrownErrors.push(a);
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
    function Js(e, t, a) {
      try {
        bh = a.source ? de(a.source) : null, d0 = de(t);
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
    function Vl(e, t, a) {
      return a = Bn(a), a.tag = t0, a.payload = { element: null }, a.callback = function() {
        he(t.source, Mo, e, t);
      }, a;
    }
    function Kt(e) {
      return e = Bn(e), e.tag = t0, e;
    }
    function er(e, t, a, i) {
      var o = a.type.getDerivedStateFromError;
      if (typeof o == "function") {
        var f = i.value;
        e.payload = function() {
          return o(f);
        }, e.callback = function() {
          Xp(a), he(
            i.source,
            Js,
            t,
            a,
            i
          );
        };
      }
      var d = a.stateNode;
      d !== null && typeof d.componentDidCatch == "function" && (e.callback = function() {
        Xp(a), he(
          i.source,
          Js,
          t,
          a,
          i
        ), typeof o != "function" && (hf === null ? hf = /* @__PURE__ */ new Set([this]) : hf.add(this)), XS(this, i), typeof o == "function" || (a.lanes & 2) === 0 && console.error(
          "%s: Error boundaries should implement getDerivedStateFromError(). In that method, return a state update to display an error message or fallback UI.",
          de(a) || "Unknown"
        );
      });
    }
    function tr(e, t, a, i, o) {
      if (a.flags |= 32768, It && qo(e, o), i !== null && typeof i == "object" && typeof i.then == "function") {
        if (t = a.alternate, t !== null && Cl(
          t,
          a,
          o,
          !0
        ), mt && (Xc = !0), a = vu.current, a !== null) {
          switch (a.tag) {
            case 13:
              return Vi === null ? hd() : a.alternate === null && il === Jc && (il = p0), a.flags &= -257, a.flags |= 65536, a.lanes = o, i === e0 ? a.flags |= 16384 : (t = a.updateQueue, t === null ? a.updateQueue = /* @__PURE__ */ new Set([i]) : t.add(i), Ky(e, i, o)), !1;
            case 22:
              return a.flags |= 65536, i === e0 ? a.flags |= 16384 : (t = a.updateQueue, t === null ? (t = {
                transitions: null,
                markerInstances: null,
                retryQueue: /* @__PURE__ */ new Set([i])
              }, a.updateQueue = t) : (a = t.retryQueue, a === null ? t.retryQueue = /* @__PURE__ */ new Set([i]) : a.add(i)), Ky(e, i, o)), !1;
          }
          throw Error(
            "Unexpected Suspense handler tag (" + a.tag + "). This is a bug in React."
          );
        }
        return Ky(e, i, o), hd(), !1;
      }
      if (mt)
        return Xc = !0, t = vu.current, t !== null ? ((t.flags & 65536) === 0 && (t.flags |= 256), t.flags |= 65536, t.lanes = o, i !== Wg && so(
          Oa(
            Error(
              "There was an error while hydrating but React was able to recover by instead client rendering from the nearest Suspense boundary.",
              { cause: i }
            ),
            a
          )
        )) : (i !== Wg && so(
          Oa(
            Error(
              "There was an error while hydrating but React was able to recover by instead client rendering the entire root.",
              { cause: i }
            ),
            a
          )
        ), e = e.current.alternate, e.flags |= 65536, o &= -o, e.lanes |= o, i = Oa(i, a), o = Vl(
          e.stateNode,
          i,
          o
        ), yo(e, o), il !== $r && (il = Rh)), !1;
      var f = Oa(
        Error(
          "There was an error during concurrent rendering but React was able to recover by instead synchronously rendering the entire root.",
          { cause: i }
        ),
        a
      );
      if (sp === null ? sp = [f] : sp.push(f), il !== $r && (il = Rh), t === null) return !0;
      i = Oa(i, a), a = t;
      do {
        switch (a.tag) {
          case 3:
            return a.flags |= 65536, e = o & -o, a.lanes |= e, e = Vl(
              a.stateNode,
              i,
              e
            ), yo(a, e), !1;
          case 1:
            if (t = a.type, f = a.stateNode, (a.flags & 128) === 0 && (typeof t.getDerivedStateFromError == "function" || f !== null && typeof f.componentDidCatch == "function" && (hf === null || !hf.has(f))))
              return a.flags |= 65536, o &= -o, a.lanes |= o, o = Kt(o), er(
                o,
                e,
                a,
                i
              ), yo(a, o), !1;
        }
        a = a.return;
      } while (a !== null);
      return !1;
    }
    function nl(e, t, a, i) {
      t.child = e === null ? j1(t, null, a, i) : vh(
        t,
        e.child,
        a,
        i
      );
    }
    function ks(e, t, a, i, o) {
      a = a.render;
      var f = t.ref;
      if ("ref" in i) {
        var d = {};
        for (var h in i)
          h !== "ref" && (d[h] = i[h]);
      } else d = i;
      return ri(t), Bt(t), i = mi(
        e,
        t,
        a,
        d,
        f,
        o
      ), h = ra(), ua(), e !== null && !Jl ? (Mu(e, t, o), Kn(e, t, o)) : (mt && h && Ds(t), t.flags |= 1, nl(e, t, i, o), t.child);
    }
    function Zn(e, t, a, i, o) {
      if (e === null) {
        var f = a.type;
        return typeof f == "function" && !$h(f) && f.defaultProps === void 0 && a.compare === null ? (a = nc(f), t.tag = 15, t.type = a, Is(t, f), lr(
          e,
          t,
          a,
          i,
          o
        )) : (e = Os(
          a.type,
          null,
          i,
          t,
          t.mode,
          o
        ), e.ref = t.ref, e.return = t, t.child = e);
      }
      if (f = e.child, !nd(e, o)) {
        var d = f.memoizedProps;
        if (a = a.compare, a = a !== null ? a : Hf, a(d, i) && e.ref === t.ref)
          return Kn(
            e,
            t,
            o
          );
      }
      return t.flags |= 1, e = Hn(f, i), e.ref = t.ref, e.return = t, t.child = e;
    }
    function lr(e, t, a, i, o) {
      if (e !== null) {
        var f = e.memoizedProps;
        if (Hf(f, i) && e.ref === t.ref && t.type === e.type)
          if (Jl = !1, t.pendingProps = i = f, nd(e, o))
            (e.flags & 131072) !== 0 && (Jl = !0);
          else
            return t.lanes = e.lanes, Kn(e, t, o);
      }
      return Fs(
        e,
        t,
        a,
        i,
        o
      );
    }
    function $s(e, t, a) {
      var i = t.pendingProps, o = i.children, f = e !== null ? e.memoizedState : null;
      if (i.mode === "hidden") {
        if ((t.flags & 128) !== 0) {
          if (i = f !== null ? f.baseLanes | a : a, e !== null) {
            for (o = t.child = e.child, f = 0; o !== null; )
              f = f | o.lanes | o.childLanes, o = o.sibling;
            t.childLanes = f & ~i;
          } else t.childLanes = 0, t.child = null;
          return Ws(
            e,
            t,
            i,
            a
          );
        }
        if ((a & 536870912) !== 0)
          t.memoizedState = { baseLanes: 0, cachePool: null }, e !== null && Us(
            t,
            f !== null ? f.cachePool : null
          ), f !== null ? fa(t, f) : Lf(t), bi(t);
        else
          return t.lanes = t.childLanes = 536870912, Ws(
            e,
            t,
            f !== null ? f.baseLanes | a : a,
            a
          );
      } else
        f !== null ? (Us(t, f.cachePool), fa(t, f), gn(t), t.memoizedState = null) : (e !== null && Us(t, null), Lf(t), gn(t));
      return nl(e, t, o, a), t.child;
    }
    function Ws(e, t, a, i) {
      var o = cy();
      return o = o === null ? null : {
        parent: Yl._currentValue,
        pool: o
      }, t.memoizedState = {
        baseLanes: a,
        cachePool: o
      }, e !== null && Us(t, null), Lf(t), bi(t), e !== null && Cl(e, t, i, !0), null;
    }
    function ar(e, t) {
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
    function Fs(e, t, a, i, o) {
      if (a.prototype && typeof a.prototype.render == "function") {
        var f = je(a) || "Unknown";
        I1[f] || (console.error(
          "The <%s /> component appears to have a render method, but doesn't extend React.Component. This is likely to cause errors. Change %s to extend React.Component instead.",
          f,
          f
        ), I1[f] = !0);
      }
      return t.mode & Ta && $u.recordLegacyContextWarning(
        t,
        null
      ), e === null && (Is(t, t.type), a.contextTypes && (f = je(a) || "Unknown", eb[f] || (eb[f] = !0, console.error(
        "%s uses the legacy contextTypes API which was removed in React 19. Use React.createContext() with React.useContext() instead. (https://react.dev/link/legacy-context)",
        f
      )))), ri(t), Bt(t), a = mi(
        e,
        t,
        a,
        i,
        void 0,
        o
      ), i = ra(), ua(), e !== null && !Jl ? (Mu(e, t, o), Kn(e, t, o)) : (mt && i && Ds(t), t.flags |= 1, nl(e, t, a, o), t.child);
    }
    function Ey(e, t, a, i, o, f) {
      return ri(t), Bt(t), Zc = -1, lp = e !== null && e.type !== t.type, t.updateQueue = null, a = go(
        t,
        i,
        a,
        o
      ), Vf(e, t), i = ra(), ua(), e !== null && !Jl ? (Mu(e, t, f), Kn(e, t, f)) : (mt && i && Ds(t), t.flags |= 1, nl(e, t, a, f), t.child);
    }
    function Ry(e, t, a, i, o) {
      switch (Ne(t)) {
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
          if (t.lanes |= h, d = wt, d === null)
            throw Error(
              "Expected a work-in-progress root. This is a bug in React. Please file an issue."
            );
          h = Kt(h), er(
            h,
            d,
            t,
            Oa(f, t)
          ), yo(t, h);
      }
      if (ri(t), t.stateNode === null) {
        if (d = uf, f = a.contextType, "contextType" in a && f !== null && (f === void 0 || f.$$typeof !== Pa) && !$1.has(a) && ($1.add(a), h = f === void 0 ? " However, it is set to undefined. This can be caused by a typo or by mixing up named and default imports. This can also happen due to a circular dependency, so try moving the createContext() call to a separate file." : typeof f != "object" ? " However, it is set to a " + typeof f + "." : f.$$typeof === Gd ? " Did you accidentally pass the Context.Consumer instead?" : " However, it is set to an object with keys {" + Object.keys(f).join(", ") + "}.", console.error(
          "%s defines an invalid contextType. contextType should point to the Context object returned by React.createContext().%s",
          je(a) || "Component",
          h
        )), typeof f == "object" && f !== null && (d = Nt(f)), f = new a(i, d), t.mode & Ta) {
          oe(!0);
          try {
            f = new a(i, d);
          } finally {
            oe(!1);
          }
        }
        if (d = t.memoizedState = f.state !== null && f.state !== void 0 ? f.state : null, f.updater = r0, t.stateNode = f, f._reactInternals = t, f._reactInternalInstance = G1, typeof a.getDerivedStateFromProps == "function" && d === null && (d = je(a) || "Component", V1.has(d) || (V1.add(d), console.error(
          "`%s` uses `getDerivedStateFromProps` but its initial state is %s. This is not recommended. Instead, define the initial state by assigning an object to `this.state` in the constructor of `%s`. This ensures that `getDerivedStateFromProps` arguments have a consistent shape.",
          d,
          f.state === null ? "null" : "undefined",
          d
        ))), typeof a.getDerivedStateFromProps == "function" || typeof f.getSnapshotBeforeUpdate == "function") {
          var v = h = d = null;
          if (typeof f.componentWillMount == "function" && f.componentWillMount.__suppressDeprecationWarning !== !0 ? d = "componentWillMount" : typeof f.UNSAFE_componentWillMount == "function" && (d = "UNSAFE_componentWillMount"), typeof f.componentWillReceiveProps == "function" && f.componentWillReceiveProps.__suppressDeprecationWarning !== !0 ? h = "componentWillReceiveProps" : typeof f.UNSAFE_componentWillReceiveProps == "function" && (h = "UNSAFE_componentWillReceiveProps"), typeof f.componentWillUpdate == "function" && f.componentWillUpdate.__suppressDeprecationWarning !== !0 ? v = "componentWillUpdate" : typeof f.UNSAFE_componentWillUpdate == "function" && (v = "UNSAFE_componentWillUpdate"), d !== null || h !== null || v !== null) {
            f = je(a) || "Component";
            var b = typeof a.getDerivedStateFromProps == "function" ? "getDerivedStateFromProps()" : "getSnapshotBeforeUpdate()";
            Q1.has(f) || (Q1.add(f), console.error(
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
        f = t.stateNode, d = je(a) || "Component", f.render || (a.prototype && typeof a.prototype.render == "function" ? console.error(
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
        ), a.childContextTypes && !k1.has(a) && (k1.add(a), console.error(
          "%s uses the legacy childContextTypes API which was removed in React 19. Use React.createContext() instead. (https://react.dev/link/legacy-context)",
          d
        )), a.contextTypes && !J1.has(a) && (J1.add(a), console.error(
          "%s uses the legacy contextTypes API which was removed in React 19. Use React.createContext() with static contextType instead. (https://react.dev/link/legacy-context)",
          d
        )), typeof f.componentShouldUpdate == "function" && console.error(
          "%s has a method called componentShouldUpdate(). Did you mean shouldComponentUpdate()? The name is phrased as a question because the function is expected to return a value.",
          d
        ), a.prototype && a.prototype.isPureReactComponent && typeof f.shouldComponentUpdate < "u" && console.error(
          "%s has a method called shouldComponentUpdate(). shouldComponentUpdate should not be used when extending React.PureComponent. Please extend React.Component if shouldComponentUpdate is used.",
          je(a) || "A pure component"
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
        ), typeof f.getSnapshotBeforeUpdate != "function" || typeof f.componentDidUpdate == "function" || X1.has(a) || (X1.add(a), console.error(
          "%s: getSnapshotBeforeUpdate() should be used with componentDidUpdate(). This component defines getSnapshotBeforeUpdate() only.",
          je(a)
        )), typeof f.getDerivedStateFromProps == "function" && console.error(
          "%s: getDerivedStateFromProps() is defined as an instance method and will be ignored. Instead, declare it as a static method.",
          d
        ), typeof f.getDerivedStateFromError == "function" && console.error(
          "%s: getDerivedStateFromError() is defined as an instance method and will be ignored. Instead, declare it as a static method.",
          d
        ), typeof a.getSnapshotBeforeUpdate == "function" && console.error(
          "%s: getSnapshotBeforeUpdate() is defined as a static method and will be ignored. Instead, declare it as an instance method.",
          d
        ), (h = f.state) && (typeof h != "object" || qe(h)) && console.error("%s.state: must be set to an object or null", d), typeof f.getChildContext == "function" && typeof a.childContextTypes != "object" && console.error(
          "%s.getChildContext(): childContextTypes must be defined in order to use getChildContext().",
          d
        ), f = t.stateNode, f.props = i, f.state = t.memoizedState, f.refs = {}, oa(t), d = a.contextType, f.context = typeof d == "object" && d !== null ? Nt(d) : uf, f.state === i && (d = je(a) || "Component", Z1.has(d) || (Z1.add(d), console.error(
          "%s: It is not recommended to assign props directly to state because updates to props won't be reflected in state. In most cases, it is better to use props directly.",
          d
        ))), t.mode & Ta && $u.recordLegacyContextWarning(
          t,
          f
        ), $u.recordUnsafeLifecycleWarnings(
          t,
          f
        ), f.state = t.memoizedState, d = a.getDerivedStateFromProps, typeof d == "function" && (Qt(
          t,
          a,
          d,
          i
        ), f.state = t.memoizedState), typeof a.getDerivedStateFromProps == "function" || typeof f.getSnapshotBeforeUpdate == "function" || typeof f.UNSAFE_componentWillMount != "function" && typeof f.componentWillMount != "function" || (d = f.state, typeof f.componentWillMount == "function" && f.componentWillMount(), typeof f.UNSAFE_componentWillMount == "function" && f.UNSAFE_componentWillMount(), d !== f.state && (console.error(
          "%s.componentWillMount(): Assigning directly to this.state is deprecated (except inside a component's constructor). Use setState instead.",
          de(t) || "Component"
        ), r0.enqueueReplaceState(
          f,
          f.state,
          null
        )), mo(t, i, f, o), Yn(), f.state = t.memoizedState), typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & ku) !== Gt && (t.flags |= 134217728), f = !0;
      } else if (e === null) {
        f = t.stateNode;
        var q = t.memoizedProps;
        h = Si(a, q), f.props = h;
        var L = f.context;
        v = a.contextType, d = uf, typeof v == "object" && v !== null && (d = Nt(v)), b = a.getDerivedStateFromProps, v = typeof b == "function" || typeof f.getSnapshotBeforeUpdate == "function", q = t.pendingProps !== q, v || typeof f.UNSAFE_componentWillReceiveProps != "function" && typeof f.componentWillReceiveProps != "function" || (q || L !== d) && Zs(
          t,
          f,
          i,
          d
        ), cf = !1;
        var x = t.memoizedState;
        f.state = x, mo(t, i, f, o), Yn(), L = t.memoizedState, q || x !== L || cf ? (typeof b == "function" && (Qt(
          t,
          a,
          b,
          i
        ), L = t.memoizedState), (h = cf || Qs(
          t,
          a,
          h,
          i,
          x,
          L,
          d
        )) ? (v || typeof f.UNSAFE_componentWillMount != "function" && typeof f.componentWillMount != "function" || (typeof f.componentWillMount == "function" && f.componentWillMount(), typeof f.UNSAFE_componentWillMount == "function" && f.UNSAFE_componentWillMount()), typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & ku) !== Gt && (t.flags |= 134217728)) : (typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & ku) !== Gt && (t.flags |= 134217728), t.memoizedProps = i, t.memoizedState = L), f.props = i, f.state = L, f.context = d, f = h) : (typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & ku) !== Gt && (t.flags |= 134217728), f = !1);
      } else {
        f = t.stateNode, di(e, t), d = t.memoizedProps, v = Si(a, d), f.props = v, b = t.pendingProps, x = f.context, L = a.contextType, h = uf, typeof L == "object" && L !== null && (h = Nt(L)), q = a.getDerivedStateFromProps, (L = typeof q == "function" || typeof f.getSnapshotBeforeUpdate == "function") || typeof f.UNSAFE_componentWillReceiveProps != "function" && typeof f.componentWillReceiveProps != "function" || (d !== b || x !== h) && Zs(
          t,
          f,
          i,
          h
        ), cf = !1, x = t.memoizedState, f.state = x, mo(t, i, f, o), Yn();
        var V = t.memoizedState;
        d !== b || x !== V || cf || e !== null && e.dependencies !== null && fi(e.dependencies) ? (typeof q == "function" && (Qt(
          t,
          a,
          q,
          i
        ), V = t.memoizedState), (v = cf || Qs(
          t,
          a,
          v,
          i,
          x,
          V,
          h
        ) || e !== null && e.dependencies !== null && fi(e.dependencies)) ? (L || typeof f.UNSAFE_componentWillUpdate != "function" && typeof f.componentWillUpdate != "function" || (typeof f.componentWillUpdate == "function" && f.componentWillUpdate(i, V, h), typeof f.UNSAFE_componentWillUpdate == "function" && f.UNSAFE_componentWillUpdate(
          i,
          V,
          h
        )), typeof f.componentDidUpdate == "function" && (t.flags |= 4), typeof f.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof f.componentDidUpdate != "function" || d === e.memoizedProps && x === e.memoizedState || (t.flags |= 4), typeof f.getSnapshotBeforeUpdate != "function" || d === e.memoizedProps && x === e.memoizedState || (t.flags |= 1024), t.memoizedProps = i, t.memoizedState = V), f.props = i, f.state = V, f.context = h, f = v) : (typeof f.componentDidUpdate != "function" || d === e.memoizedProps && x === e.memoizedState || (t.flags |= 4), typeof f.getSnapshotBeforeUpdate != "function" || d === e.memoizedProps && x === e.memoizedState || (t.flags |= 1024), f = !1);
      }
      if (h = f, ar(e, t), d = (t.flags & 128) !== 0, h || d) {
        if (h = t.stateNode, Rf(t), d && typeof a.getDerivedStateFromError != "function")
          a = null, ln = -1;
        else {
          if (Bt(t), a = O1(h), t.mode & Ta) {
            oe(!0);
            try {
              O1(h);
            } finally {
              oe(!1);
            }
          }
          ua();
        }
        t.flags |= 1, e !== null && d ? (t.child = vh(
          t,
          e.child,
          null,
          o
        ), t.child = vh(
          t,
          null,
          a,
          o
        )) : nl(e, t, a, o), t.memoizedState = h.state, e = t.child;
      } else
        e = Kn(
          e,
          t,
          o
        );
      return o = t.stateNode, f && o.props !== i && (Sh || console.error(
        "It looks like %s is reassigning its own `this.props` while rendering. This is not supported and can lead to confusing bugs.",
        de(t) || "a component"
      ), Sh = !0), e;
    }
    function Ay(e, t, a, i) {
      return cc(), t.flags |= 256, nl(e, t, a, i), t.child;
    }
    function Is(e, t) {
      t && t.childContextTypes && console.error(
        `childContextTypes cannot be defined on a function component.
  %s.childContextTypes = ...`,
        t.displayName || t.name || "Component"
      ), typeof t.getDerivedStateFromProps == "function" && (e = je(t) || "Unknown", tb[e] || (console.error(
        "%s: Function components do not support getDerivedStateFromProps.",
        e
      ), tb[e] = !0)), typeof t.contextType == "object" && t.contextType !== null && (t = je(t) || "Unknown", P1[t] || (console.error(
        "%s: Function components do not support contextType.",
        t
      ), P1[t] = !0));
    }
    function nr(e) {
      return { baseLanes: e, cachePool: Jp() };
    }
    function Ps(e, t, a) {
      return e = e !== null ? e.childLanes & ~a : 0, t && (e |= Dn), e;
    }
    function Ip(e, t, a) {
      var i, o = t.pendingProps;
      Ae(t) && (t.flags |= 128);
      var f = !1, d = (t.flags & 128) !== 0;
      if ((i = d) || (i = e !== null && e.memoizedState === null ? !1 : (Gl.current & np) !== 0), i && (f = !0, t.flags &= -129), i = (t.flags & 32) !== 0, t.flags &= -33, e === null) {
        if (mt) {
          if (f ? za(t) : gn(t), mt) {
            var h = ul, v;
            if (!(v = !h)) {
              e: {
                var b = h;
                for (v = Gi; b.nodeType !== 8; ) {
                  if (!v) {
                    v = null;
                    break e;
                  }
                  if (b = wl(b.nextSibling), b === null) {
                    v = null;
                    break e;
                  }
                }
                v = b;
              }
              v !== null ? (rn(), t.memoizedState = {
                dehydrated: v,
                treeContext: Vr !== null ? { id: Lc, overflow: Vc } : null,
                retryLane: 536870912,
                hydrationErrors: null
              }, b = D(18, null, null, Gt), b.stateNode = v, b.return = t, t.child = b, wa = t, ul = null, v = !0) : v = !1, v = !v;
            }
            v && (Ih(
              t,
              h
            ), Nn(t));
          }
          if (h = t.memoizedState, h !== null && (h = h.dehydrated, h !== null))
            return au(h) ? t.lanes = 32 : t.lanes = 536870912, null;
          Ma(t);
        }
        return h = o.children, o = o.fallback, f ? (gn(t), f = t.mode, h = ur(
          {
            mode: "hidden",
            children: h
          },
          f
        ), o = ii(
          o,
          f,
          a,
          null
        ), h.return = t, o.return = t, h.sibling = o, t.child = h, f = t.child, f.memoizedState = nr(a), f.childLanes = Ps(
          e,
          i,
          a
        ), t.memoizedState = y0, o) : (za(t), ed(
          t,
          h
        ));
      }
      var q = e.memoizedState;
      if (q !== null && (h = q.dehydrated, h !== null)) {
        if (d)
          t.flags & 256 ? (za(t), t.flags &= -257, t = td(
            e,
            t,
            a
          )) : t.memoizedState !== null ? (gn(t), t.child = e.child, t.flags |= 128, t = null) : (gn(t), f = o.fallback, h = t.mode, o = ur(
            {
              mode: "visible",
              children: o.children
            },
            h
          ), f = ii(
            f,
            h,
            a,
            null
          ), f.flags |= 2, o.return = t, f.return = t, o.sibling = f, t.child = o, vh(
            t,
            e.child,
            null,
            a
          ), o = t.child, o.memoizedState = nr(a), o.childLanes = Ps(
            e,
            i,
            a
          ), t.memoizedState = y0, t = f);
        else if (za(t), mt && console.error(
          "We should not be hydrating here. This is a bug in React. Please file a bug."
        ), au(h)) {
          if (i = h.nextSibling && h.nextSibling.dataset, i) {
            v = i.dgst;
            var L = i.msg;
            b = i.stck;
            var x = i.cstck;
          }
          h = L, i = v, o = b, v = f = x, f = Error(h || "The server could not finish this Suspense boundary, likely due to an error during server rendering. Switched to client rendering."), f.stack = o || "", f.digest = i, i = v === void 0 ? null : v, o = {
            value: f,
            source: null,
            stack: i
          }, typeof i == "string" && kg.set(
            f,
            o
          ), so(o), t = td(
            e,
            t,
            a
          );
        } else if (Jl || Cl(
          e,
          t,
          a,
          !1
        ), i = (a & e.childLanes) !== 0, Jl || i) {
          if (i = wt, i !== null && (o = a & -a, o = (o & 42) !== 0 ? 1 : Dl(
            o
          ), o = (o & (i.suspendedLanes | a)) !== 0 ? 0 : o, o !== 0 && o !== q.retryLane))
            throw q.retryLane = o, ca(
              e,
              o
            ), Jt(
              i,
              e,
              o
            ), F1;
          h.data === kc || hd(), t = td(
            e,
            t,
            a
          );
        } else
          h.data === kc ? (t.flags |= 192, t.child = e.child, t = null) : (e = q.treeContext, ul = wl(
            h.nextSibling
          ), wa = t, mt = !0, Xr = null, Xc = !1, hu = null, Gi = !1, e !== null && (rn(), su[du++] = Lc, su[du++] = Vc, su[du++] = Vr, Lc = e.id, Vc = e.overflow, Vr = t), t = ed(
            t,
            o.children
          ), t.flags |= 4096);
        return t;
      }
      return f ? (gn(t), f = o.fallback, h = t.mode, v = e.child, b = v.sibling, o = Hn(
        v,
        {
          mode: "hidden",
          children: o.children
        }
      ), o.subtreeFlags = v.subtreeFlags & 65011712, b !== null ? f = Hn(
        b,
        f
      ) : (f = ii(
        f,
        h,
        a,
        null
      ), f.flags |= 2), f.return = t, o.return = t, o.sibling = f, t.child = o, o = f, f = t.child, h = e.child.memoizedState, h === null ? h = nr(a) : (v = h.cachePool, v !== null ? (b = Yl._currentValue, v = v.parent !== b ? { parent: b, pool: b } : v) : v = Jp(), h = {
        baseLanes: h.baseLanes | a,
        cachePool: v
      }), f.memoizedState = h, f.childLanes = Ps(
        e,
        i,
        a
      ), t.memoizedState = y0, o) : (za(t), a = e.child, e = a.sibling, a = Hn(a, {
        mode: "visible",
        children: o.children
      }), a.return = t, a.sibling = null, e !== null && (i = t.deletions, i === null ? (t.deletions = [e], t.flags |= 16) : i.push(e)), t.child = a, t.memoizedState = null, a);
    }
    function ed(e, t) {
      return t = ur(
        { mode: "visible", children: t },
        e.mode
      ), t.return = e, e.child = t;
    }
    function ur(e, t) {
      return e = D(22, e, null, t), e.lanes = 0, e.stateNode = {
        _visibility: wv,
        _pendingMarkers: null,
        _retryCache: null,
        _transitions: null
      }, e;
    }
    function td(e, t, a) {
      return vh(t, e.child, null, a), e = ed(
        t,
        t.pendingProps.children
      ), e.flags |= 2, t.memoizedState = null, e;
    }
    function ld(e, t, a) {
      e.lanes |= t;
      var i = e.alternate;
      i !== null && (i.lanes |= t), ay(
        e.return,
        t,
        a
      );
    }
    function Oy(e, t) {
      var a = qe(e);
      return e = !a && typeof pt(e) == "function", a || e ? (a = a ? "array" : "iterable", console.error(
        "A nested %s was passed to row #%s in <SuspenseList />. Wrap it in an additional SuspenseList to configure its revealOrder: <SuspenseList revealOrder=...> ... <SuspenseList revealOrder=...>{%s}</SuspenseList> ... </SuspenseList>",
        a,
        t,
        a
      ), !1) : !0;
    }
    function ad(e, t, a, i, o) {
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
    function Dy(e, t, a) {
      var i = t.pendingProps, o = i.revealOrder, f = i.tail;
      if (i = i.children, o !== void 0 && o !== "forwards" && o !== "backwards" && o !== "together" && !lb[o])
        if (lb[o] = !0, typeof o == "string")
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
      f === void 0 || h0[f] || (f !== "collapsed" && f !== "hidden" ? (h0[f] = !0, console.error(
        '"%s" is not a supported value for tail on <SuspenseList />. Did you mean "collapsed" or "hidden"?',
        f
      )) : o !== "forwards" && o !== "backwards" && (h0[f] = !0, console.error(
        '<SuspenseList tail="%s" /> is only valid if revealOrder is "forwards" or "backwards". Did you mean to specify revealOrder="forwards"?',
        f
      )));
      e: if ((o === "forwards" || o === "backwards") && i !== void 0 && i !== null && i !== !1)
        if (qe(i)) {
          for (var d = 0; d < i.length; d++)
            if (!Oy(i[d], d)) break e;
        } else if (d = pt(i), typeof d == "function") {
          if (d = d.call(i))
            for (var h = d.next(), v = 0; !h.done; h = d.next()) {
              if (!Oy(h.value, v)) break e;
              v++;
            }
        } else
          console.error(
            'A single row was passed to a <SuspenseList revealOrder="%s" />. This is not useful since it needs multiple rows. Did you mean to pass multiple children or an array?',
            o
          );
      if (nl(e, t, i, a), i = Gl.current, (i & np) !== 0)
        i = i & gh | np, t.flags |= 128;
      else {
        if (e !== null && (e.flags & 128) !== 0)
          e: for (e = t.child; e !== null; ) {
            if (e.tag === 13)
              e.memoizedState !== null && ld(
                e,
                a,
                t
              );
            else if (e.tag === 19)
              ld(e, a, t);
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
        i &= gh;
      }
      switch (ze(Gl, i, t), o) {
        case "forwards":
          for (a = t.child, o = null; a !== null; )
            e = a.alternate, e !== null && Hu(e) === null && (o = a), a = a.sibling;
          a = o, a === null ? (o = t.child, t.child = null) : (o = a.sibling, a.sibling = null), ad(
            t,
            !1,
            o,
            a,
            f
          );
          break;
        case "backwards":
          for (a = null, o = t.child, t.child = null; o !== null; ) {
            if (e = o.alternate, e !== null && Hu(e) === null) {
              t.child = o;
              break;
            }
            e = o.sibling, o.sibling = a, a = o, o = e;
          }
          ad(
            t,
            !0,
            a,
            null,
            f
          );
          break;
        case "together":
          ad(t, !1, null, null, void 0);
          break;
        default:
          t.memoizedState = null;
      }
      return t.child;
    }
    function Kn(e, t, a) {
      if (e !== null && (t.dependencies = e.dependencies), ln = -1, sf |= t.lanes, (a & t.childLanes) === 0)
        if (e !== null) {
          if (Cl(
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
        for (e = t.child, a = Hn(e, e.pendingProps), t.child = a, a.return = t; e.sibling !== null; )
          e = e.sibling, a = a.sibling = Hn(e, e.pendingProps), a.return = t;
        a.sibling = null;
      }
      return t.child;
    }
    function nd(e, t) {
      return (e.lanes & t) !== 0 ? !0 : (e = e.dependencies, !!(e !== null && fi(e)));
    }
    function _g(e, t, a) {
      switch (t.tag) {
        case 3:
          Ht(
            t,
            t.stateNode.containerInfo
          ), oi(
            t,
            Yl,
            e.memoizedState.cache
          ), cc();
          break;
        case 27:
        case 5:
          X(t);
          break;
        case 4:
          Ht(
            t,
            t.stateNode.containerInfo
          );
          break;
        case 10:
          oi(
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
            return i.dehydrated !== null ? (za(t), t.flags |= 128, null) : (a & t.child.childLanes) !== 0 ? Ip(
              e,
              t,
              a
            ) : (za(t), e = Kn(
              e,
              t,
              a
            ), e !== null ? e.sibling : null);
          za(t);
          break;
        case 19:
          var o = (e.flags & 128) !== 0;
          if (i = (a & t.childLanes) !== 0, i || (Cl(
            e,
            t,
            a,
            !1
          ), i = (a & t.childLanes) !== 0), o) {
            if (i)
              return Dy(
                e,
                t,
                a
              );
            t.flags |= 128;
          }
          if (o = t.memoizedState, o !== null && (o.rendering = null, o.tail = null, o.lastEffect = null), ze(
            Gl,
            Gl.current,
            t
          ), i) break;
          return null;
        case 22:
        case 23:
          return t.lanes = 0, $s(e, t, a);
        case 24:
          oi(
            t,
            Yl,
            e.memoizedState.cache
          );
      }
      return Kn(e, t, a);
    }
    function ud(e, t, a) {
      if (t._debugNeedsRemount && e !== null) {
        a = Os(
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
          Jl = !0;
        else {
          if (!nd(e, a) && (t.flags & 128) === 0)
            return Jl = !1, _g(
              e,
              t,
              a
            );
          Jl = (e.flags & 131072) !== 0;
        }
      else
        Jl = !1, (i = mt) && (rn(), i = (t.flags & 1048576) !== 0), i && (i = t.index, rn(), Qp(t, Bv, i));
      switch (t.lanes = 0, t.tag) {
        case 16:
          e: if (i = t.pendingProps, e = ff(t.elementType), t.type = e, typeof e == "function")
            $h(e) ? (i = Si(
              e,
              i
            ), t.tag = 1, t.type = e = nc(e), t = Ry(
              null,
              t,
              e,
              i,
              a
            )) : (t.tag = 0, Is(t, e), t.type = e = nc(e), t = Fs(
              null,
              t,
              e,
              i,
              a
            ));
          else {
            if (e != null) {
              if (o = e.$$typeof, o === Lu) {
                t.tag = 11, t.type = e = kh(e), t = ks(
                  null,
                  t,
                  e,
                  i,
                  a
                );
                break e;
              } else if (o === _r) {
                t.tag = 14, t = Zn(
                  null,
                  t,
                  e,
                  i,
                  a
                );
                break e;
              }
            }
            throw t = "", e !== null && typeof e == "object" && e.$$typeof === xa && (t = " Did you wrap a component in React.lazy() more than once?"), e = je(e) || e, Error(
              "Element type is invalid. Received a promise that resolves to: " + e + ". Lazy element type must resolve to a class or function." + t
            );
          }
          return t;
        case 0:
          return Fs(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 1:
          return i = t.type, o = Si(
            i,
            t.pendingProps
          ), Ry(
            e,
            t,
            i,
            o,
            a
          );
        case 3:
          e: {
            if (Ht(
              t,
              t.stateNode.containerInfo
            ), e === null)
              throw Error(
                "Should have a current fiber. This is a bug in React."
              );
            i = t.pendingProps;
            var f = t.memoizedState;
            o = f.element, di(e, t), mo(t, i, null, a);
            var d = t.memoizedState;
            if (i = d.cache, oi(t, Yl, i), i !== f.cache && ny(
              t,
              [Yl],
              a,
              !0
            ), Yn(), i = d.element, f.isDehydrated)
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
                o = Oa(
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
                for (ul = wl(e.firstChild), wa = t, mt = !0, Xr = null, Xc = !1, hu = null, Gi = !0, e = j1(
                  t,
                  null,
                  i,
                  a
                ), t.child = e; e; )
                  e.flags = e.flags & -3 | 4096, e = e.sibling;
              }
            else {
              if (cc(), i === o) {
                t = Kn(
                  e,
                  t,
                  a
                );
                break e;
              }
              nl(
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
          return ar(e, t), e === null ? (e = Gu(
            t.type,
            null,
            t.pendingProps,
            null
          )) ? t.memoizedState = e : mt || (e = t.type, a = t.pendingProps, i = Dt(
            uu.current
          ), i = lt(
            i
          ).createElement(e), i[Kl] = t, i[ba] = a, $t(i, e, a), z(i), t.stateNode = i) : t.memoizedState = Gu(
            t.type,
            e.memoizedProps,
            t.pendingProps,
            e.memoizedState
          ), null;
        case 27:
          return X(t), e === null && mt && (i = Dt(uu.current), o = R(), i = t.stateNode = sm(
            t.type,
            t.pendingProps,
            i,
            o,
            !1
          ), Xc || (o = Ut(
            i,
            t.type,
            t.pendingProps,
            o
          ), o !== null && (sn(t, 0).serverProps = o)), wa = t, Gi = !0, o = ul, lu(t.type) ? (B0 = o, ul = wl(
            i.firstChild
          )) : ul = o), nl(
            e,
            t,
            t.pendingProps.children,
            a
          ), ar(e, t), e === null && (t.flags |= 4194304), t.child;
        case 5:
          return e === null && mt && (f = R(), i = ps(
            t.type,
            f.ancestorInfo
          ), o = ul, (d = !o) || (d = zi(
            o,
            t.type,
            t.pendingProps,
            Gi
          ), d !== null ? (t.stateNode = d, Xc || (f = Ut(
            d,
            t.type,
            t.pendingProps,
            f
          ), f !== null && (sn(t, 0).serverProps = f)), wa = t, ul = wl(
            d.firstChild
          ), Gi = !1, f = !0) : f = !1, d = !f), d && (i && Ih(t, o), Nn(t))), X(t), o = t.type, f = t.pendingProps, d = e !== null ? e.memoizedProps : null, i = f.children, tu(o, f) ? i = null : d !== null && tu(o, d) && (t.flags |= 32), t.memoizedState !== null && (o = mi(
            e,
            t,
            Qa,
            null,
            null,
            a
          ), bp._currentValue = o), ar(e, t), nl(
            e,
            t,
            i,
            a
          ), t.child;
        case 6:
          return e === null && mt && (e = t.pendingProps, a = R(), i = a.ancestorInfo.current, e = i != null ? _f(
            e,
            i.tag,
            a.ancestorInfo.implicitRootScope
          ) : !0, a = ul, (i = !a) || (i = Nl(
            a,
            t.pendingProps,
            Gi
          ), i !== null ? (t.stateNode = i, wa = t, ul = null, i = !0) : i = !1, i = !i), i && (e && Ih(t, a), Nn(t))), null;
        case 13:
          return Ip(e, t, a);
        case 4:
          return Ht(
            t,
            t.stateNode.containerInfo
          ), i = t.pendingProps, e === null ? t.child = vh(
            t,
            null,
            i,
            a
          ) : nl(
            e,
            t,
            i,
            a
          ), t.child;
        case 11:
          return ks(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 7:
          return nl(
            e,
            t,
            t.pendingProps,
            a
          ), t.child;
        case 8:
          return nl(
            e,
            t,
            t.pendingProps.children,
            a
          ), t.child;
        case 12:
          return t.flags |= 4, t.flags |= 2048, i = t.stateNode, i.effectDuration = -0, i.passiveEffectDuration = -0, nl(
            e,
            t,
            t.pendingProps.children,
            a
          ), t.child;
        case 10:
          return i = t.type, o = t.pendingProps, f = o.value, "value" in o || ab || (ab = !0, console.error(
            "The `value` prop is required for the `<Context.Provider>`. Did you misspell it or forget to pass it?"
          )), oi(t, i, f), nl(
            e,
            t,
            o.children,
            a
          ), t.child;
        case 9:
          return o = t.type._context, i = t.pendingProps.children, typeof i != "function" && console.error(
            "A context consumer was rendered with multiple children, or a child that isn't a function. A context consumer expects a single child that is a function. If you did pass a function, make sure there is no trailing or leading whitespace around it."
          ), ri(t), o = Nt(o), Bt(t), i = c0(
            i,
            o,
            void 0
          ), ua(), t.flags |= 1, nl(
            e,
            t,
            i,
            a
          ), t.child;
        case 14:
          return Zn(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 15:
          return lr(
            e,
            t,
            t.type,
            t.pendingProps,
            a
          );
        case 19:
          return Dy(
            e,
            t,
            a
          );
        case 31:
          return i = t.pendingProps, a = t.mode, i = {
            mode: i.mode,
            children: i.children
          }, e === null ? (e = ur(
            i,
            a
          ), e.ref = t.ref, t.child = e, e.return = t, t = e) : (e = Hn(e.child, i), e.ref = t.ref, t.child = e, e.return = t, t = e), t;
        case 22:
          return $s(e, t, a);
        case 24:
          return ri(t), i = Nt(Yl), e === null ? (o = cy(), o === null && (o = wt, f = jf(), o.pooledCache = f, oc(f), f !== null && (o.pooledCacheLanes |= a), o = f), t.memoizedState = {
            parent: i,
            cache: o
          }, oa(t), oi(t, Yl, o)) : ((e.lanes & a) !== 0 && (di(e, t), mo(t, null, null, a), Yn()), o = e.memoizedState, f = t.memoizedState, o.parent !== i ? (o = {
            parent: i,
            cache: i
          }, t.memoizedState = o, t.lanes === 0 && (t.memoizedState = t.updateQueue.baseState = o), oi(t, Yl, i)) : (i = f.cache, oi(t, Yl, i), i !== o.cache && ny(
            t,
            [Yl],
            a,
            !0
          ))), nl(
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
    function da(e) {
      e.flags |= 4;
    }
    function ir(e, t) {
      if (t.type !== "stylesheet" || (t.state.loading & gu) !== ns)
        e.flags &= -16777217;
      else if (e.flags |= 16777216, !Sr(t)) {
        if (t = vu.current, t !== null && ((nt & 4194048) === nt ? Vi !== null : (nt & 62914560) !== nt && (nt & 536870912) === 0 || t !== Vi))
          throw ep = e0, h1;
        e.flags |= 8192;
      }
    }
    function cr(e, t) {
      t !== null && (e.flags |= 4), e.flags & 16384 && (t = e.tag !== 22 ? Un() : 536870912, e.lanes |= t, Ir |= t);
    }
    function Ti(e, t) {
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
    function zt(e) {
      var t = e.alternate !== null && e.alternate.child === e.child, a = 0, i = 0;
      if (t)
        if ((e.mode & la) !== Gt) {
          for (var o = e.selfBaseDuration, f = e.child; f !== null; )
            a |= f.lanes | f.childLanes, i |= f.subtreeFlags & 65011712, i |= f.flags & 65011712, o += f.treeBaseDuration, f = f.sibling;
          e.treeBaseDuration = o;
        } else
          for (o = e.child; o !== null; )
            a |= o.lanes | o.childLanes, i |= o.subtreeFlags & 65011712, i |= o.flags & 65011712, o.return = e, o = o.sibling;
      else if ((e.mode & la) !== Gt) {
        o = e.actualDuration, f = e.selfBaseDuration;
        for (var d = e.child; d !== null; )
          a |= d.lanes | d.childLanes, i |= d.subtreeFlags, i |= d.flags, o += d.actualDuration, f += d.treeBaseDuration, d = d.sibling;
        e.actualDuration = o, e.treeBaseDuration = f;
      } else
        for (o = e.child; o !== null; )
          a |= o.lanes | o.childLanes, i |= o.subtreeFlags, i |= o.flags, o.return = e, o = o.sibling;
      return e.subtreeFlags |= i, e.childLanes = a, t;
    }
    function Pp(e, t, a) {
      var i = t.pendingProps;
      switch (zs(t), t.tag) {
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
          return zt(t), null;
        case 1:
          return zt(t), null;
        case 3:
          return a = t.stateNode, i = null, e !== null && (i = e.memoizedState.cache), t.memoizedState.cache !== i && (t.flags |= 2048), Du(Yl, t), le(t), a.pendingContext && (a.context = a.pendingContext, a.pendingContext = null), (e === null || e.child === null) && (ic(t) ? (ly(), da(t)) : e === null || e.memoizedState.isDehydrated && (t.flags & 256) === 0 || (t.flags |= 1024, ty())), zt(t), null;
        case 26:
          return a = t.memoizedState, e === null ? (da(t), a !== null ? (zt(t), ir(
            t,
            a
          )) : (zt(t), t.flags &= -16777217)) : a ? a !== e.memoizedState ? (da(t), zt(t), ir(
            t,
            a
          )) : (zt(t), t.flags &= -16777217) : (e.memoizedProps !== i && da(t), zt(t), t.flags &= -16777217), null;
        case 27:
          I(t), a = Dt(uu.current);
          var o = t.type;
          if (e !== null && t.stateNode != null)
            e.memoizedProps !== i && da(t);
          else {
            if (!i) {
              if (t.stateNode === null)
                throw Error(
                  "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
                );
              return zt(t), null;
            }
            e = R(), ic(t) ? Ph(t) : (e = sm(
              o,
              i,
              a,
              e,
              !0
            ), t.stateNode = e, da(t));
          }
          return zt(t), null;
        case 5:
          if (I(t), a = t.type, e !== null && t.stateNode != null)
            e.memoizedProps !== i && da(t);
          else {
            if (!i) {
              if (t.stateNode === null)
                throw Error(
                  "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
                );
              return zt(t), null;
            }
            if (o = R(), ic(t))
              Ph(t);
            else {
              switch (e = Dt(uu.current), ps(a, o.ancestorInfo), o = o.context, e = lt(e), o) {
                case _h:
                  e = e.createElementNS(nf, a);
                  break;
                case rg:
                  e = e.createElementNS(
                    Gr,
                    a
                  );
                  break;
                default:
                  switch (a) {
                    case "svg":
                      e = e.createElementNS(
                        nf,
                        a
                      );
                      break;
                    case "math":
                      e = e.createElementNS(
                        Gr,
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
                      ), Object.prototype.toString.call(e) !== "[object HTMLUnknownElement]" || Xu.call(
                        Eb,
                        a
                      ) || (Eb[a] = !0, console.error(
                        "The tag <%s> is unrecognized in this browser. If you meant to render a React component, start its name with an uppercase letter.",
                        a
                      )));
                  }
              }
              e[Kl] = t, e[ba] = i;
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
              e: switch ($t(e, a, i), a) {
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
              e && da(t);
            }
          }
          return zt(t), t.flags &= -16777217, null;
        case 6:
          if (e && t.stateNode != null)
            e.memoizedProps !== i && da(t);
          else {
            if (typeof i != "string" && t.stateNode === null)
              throw Error(
                "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
              );
            if (e = Dt(uu.current), a = R(), ic(t)) {
              e = t.stateNode, a = t.memoizedProps, o = !Xc, i = null;
              var f = wa;
              if (f !== null)
                switch (f.tag) {
                  case 3:
                    o && (o = Cd(
                      e,
                      a,
                      i
                    ), o !== null && (sn(t, 0).serverProps = o));
                    break;
                  case 27:
                  case 5:
                    i = f.memoizedProps, o && (o = Cd(
                      e,
                      a,
                      i
                    ), o !== null && (sn(
                      t,
                      0
                    ).serverProps = o));
                }
              e[Kl] = t, e = !!(e.nodeValue === a || i !== null && i.suppressHydrationWarning === !0 || tm(e.nodeValue, a)), e || Nn(t);
            } else
              o = a.ancestorInfo.current, o != null && _f(
                i,
                o.tag,
                a.ancestorInfo.implicitRootScope
              ), e = lt(e).createTextNode(
                i
              ), e[Kl] = t, t.stateNode = e;
          }
          return zt(t), null;
        case 13:
          if (i = t.memoizedState, e === null || e.memoizedState !== null && e.memoizedState.dehydrated !== null) {
            if (o = ic(t), i !== null && i.dehydrated !== null) {
              if (e === null) {
                if (!o)
                  throw Error(
                    "A dehydrated suspense component was completed without a hydrated node. This is probably a bug in React."
                  );
                if (o = t.memoizedState, o = o !== null ? o.dehydrated : null, !o)
                  throw Error(
                    "Expected to have a hydrated suspense instance. This error is likely caused by a bug in React. Please file an issue."
                  );
                o[Kl] = t, zt(t), (t.mode & la) !== Gt && i !== null && (o = t.child, o !== null && (t.treeBaseDuration -= o.treeBaseDuration));
              } else
                ly(), cc(), (t.flags & 128) === 0 && (t.memoizedState = null), t.flags |= 4, zt(t), (t.mode & la) !== Gt && i !== null && (o = t.child, o !== null && (t.treeBaseDuration -= o.treeBaseDuration));
              o = !1;
            } else
              o = ty(), e !== null && e.memoizedState !== null && (e.memoizedState.hydrationErrors = o), o = !0;
            if (!o)
              return t.flags & 256 ? (Ma(t), t) : (Ma(t), null);
          }
          return Ma(t), (t.flags & 128) !== 0 ? (t.lanes = a, (t.mode & la) !== Gt && qn(t), t) : (a = i !== null, e = e !== null && e.memoizedState !== null, a && (i = t.child, o = null, i.alternate !== null && i.alternate.memoizedState !== null && i.alternate.memoizedState.cachePool !== null && (o = i.alternate.memoizedState.cachePool.pool), f = null, i.memoizedState !== null && i.memoizedState.cachePool !== null && (f = i.memoizedState.cachePool.pool), f !== o && (i.flags |= 2048)), a !== e && a && (t.child.flags |= 8192), cr(t, t.updateQueue), zt(t), (t.mode & la) !== Gt && a && (e = t.child, e !== null && (t.treeBaseDuration -= e.treeBaseDuration)), null);
        case 4:
          return le(t), e === null && Py(
            t.stateNode.containerInfo
          ), zt(t), null;
        case 10:
          return Du(t.type, t), zt(t), null;
        case 19:
          if (ve(Gl, t), o = t.memoizedState, o === null) return zt(t), null;
          if (i = (t.flags & 128) !== 0, f = o.rendering, f === null)
            if (i) Ti(o, !1);
            else {
              if (il !== Jc || e !== null && (e.flags & 128) !== 0)
                for (e = t.child; e !== null; ) {
                  if (f = Hu(e), f !== null) {
                    for (t.flags |= 128, Ti(o, !1), e = f.updateQueue, t.updateQueue = e, cr(t, e), t.subtreeFlags = 0, e = a, a = t.child; a !== null; )
                      Wh(a, e), a = a.sibling;
                    return ze(
                      Gl,
                      Gl.current & gh | np,
                      t
                    ), t.child;
                  }
                  e = e.sibling;
                }
              o.tail !== null && iu() > Iv && (t.flags |= 128, i = !0, Ti(o, !1), t.lanes = 4194304);
            }
          else {
            if (!i)
              if (e = Hu(f), e !== null) {
                if (t.flags |= 128, i = !0, e = e.updateQueue, t.updateQueue = e, cr(t, e), Ti(o, !0), o.tail === null && o.tailMode === "hidden" && !f.alternate && !mt)
                  return zt(t), null;
              } else
                2 * iu() - o.renderingStartTime > Iv && a !== 536870912 && (t.flags |= 128, i = !0, Ti(o, !1), t.lanes = 4194304);
            o.isBackwards ? (f.sibling = t.child, t.child = f) : (e = o.last, e !== null ? e.sibling = f : t.child = f, o.last = f);
          }
          return o.tail !== null ? (e = o.tail, o.rendering = e, o.tail = e.sibling, o.renderingStartTime = iu(), e.sibling = null, a = Gl.current, a = i ? a & gh | np : a & gh, ze(Gl, a, t), e) : (zt(t), null);
        case 22:
        case 23:
          return Ma(t), mn(t), i = t.memoizedState !== null, e !== null ? e.memoizedState !== null !== i && (t.flags |= 8192) : i && (t.flags |= 8192), i ? (a & 536870912) !== 0 && (t.flags & 128) === 0 && (zt(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : zt(t), a = t.updateQueue, a !== null && cr(t, a.retryQueue), a = null, e !== null && e.memoizedState !== null && e.memoizedState.cachePool !== null && (a = e.memoizedState.cachePool.pool), i = null, t.memoizedState !== null && t.memoizedState.cachePool !== null && (i = t.memoizedState.cachePool.pool), i !== a && (t.flags |= 2048), e !== null && ve(Kr, t), null;
        case 24:
          return a = null, e !== null && (a = e.memoizedState.cache), t.memoizedState.cache !== a && (t.flags |= 2048), Du(Yl, t), zt(t), null;
        case 25:
          return null;
        case 30:
          return null;
      }
      throw Error(
        "Unknown unit of work tag (" + t.tag + "). This error is likely caused by a bug in React. Please file an issue."
      );
    }
    function ev(e, t) {
      switch (zs(t), t.tag) {
        case 1:
          return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & la) !== Gt && qn(t), t) : null;
        case 3:
          return Du(Yl, t), le(t), e = t.flags, (e & 65536) !== 0 && (e & 128) === 0 ? (t.flags = e & -65537 | 128, t) : null;
        case 26:
        case 27:
        case 5:
          return I(t), null;
        case 13:
          if (Ma(t), e = t.memoizedState, e !== null && e.dehydrated !== null) {
            if (t.alternate === null)
              throw Error(
                "Threw in newly mounted dehydrated component. This is likely a bug in React. Please file an issue."
              );
            cc();
          }
          return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & la) !== Gt && qn(t), t) : null;
        case 19:
          return ve(Gl, t), null;
        case 4:
          return le(t), null;
        case 10:
          return Du(t.type, t), null;
        case 22:
        case 23:
          return Ma(t), mn(t), e !== null && ve(Kr, t), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & la) !== Gt && qn(t), t) : null;
        case 24:
          return Du(Yl, t), null;
        case 25:
          return null;
        default:
          return null;
      }
    }
    function zy(e, t) {
      switch (zs(t), t.tag) {
        case 3:
          Du(Yl, t), le(t);
          break;
        case 26:
        case 27:
        case 5:
          I(t);
          break;
        case 4:
          le(t);
          break;
        case 13:
          Ma(t);
          break;
        case 19:
          ve(Gl, t);
          break;
        case 10:
          Du(t.type, t);
          break;
        case 22:
        case 23:
          Ma(t), mn(t), e !== null && ve(Kr, t);
          break;
        case 24:
          Du(Yl, t);
      }
    }
    function bn(e) {
      return (e.mode & la) !== Gt;
    }
    function My(e, t) {
      bn(e) ? (hn(), vc(t, e), La()) : vc(t, e);
    }
    function id(e, t, a) {
      bn(e) ? (hn(), gc(
        a,
        e,
        t
      ), La()) : gc(
        a,
        e,
        t
      );
    }
    function vc(e, t) {
      try {
        var a = t.updateQueue, i = a !== null ? a.lastEffect : null;
        if (i !== null) {
          var o = i.next;
          a = o;
          do {
            if ((a.tag & e) === e && ((e & jl) !== yu ? fe !== null && typeof fe.markComponentPassiveEffectMountStarted == "function" && fe.markComponentPassiveEffectMountStarted(
              t
            ) : (e & aa) !== yu && fe !== null && typeof fe.markComponentLayoutEffectMountStarted == "function" && fe.markComponentLayoutEffectMountStarted(
              t
            ), i = void 0, (e & qa) !== yu && (zh = !0), i = he(
              t,
              QS,
              a
            ), (e & qa) !== yu && (zh = !1), (e & jl) !== yu ? fe !== null && typeof fe.markComponentPassiveEffectMountStopped == "function" && fe.markComponentPassiveEffectMountStopped() : (e & aa) !== yu && fe !== null && typeof fe.markComponentLayoutEffectMountStopped == "function" && fe.markComponentLayoutEffectMountStopped(), i !== void 0 && typeof i != "function")) {
              var f = void 0;
              f = (a.tag & aa) !== 0 ? "useLayoutEffect" : (a.tag & qa) !== 0 ? "useInsertionEffect" : "useEffect";
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

Learn more about data fetching with Hooks: https://react.dev/link/hooks-data-fetching` : " You returned: " + i, he(
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
    function gc(e, t, a) {
      try {
        var i = t.updateQueue, o = i !== null ? i.lastEffect : null;
        if (o !== null) {
          var f = o.next;
          i = f;
          do {
            if ((i.tag & e) === e) {
              var d = i.inst, h = d.destroy;
              h !== void 0 && (d.destroy = void 0, (e & jl) !== yu ? fe !== null && typeof fe.markComponentPassiveEffectUnmountStarted == "function" && fe.markComponentPassiveEffectUnmountStarted(
                t
              ) : (e & aa) !== yu && fe !== null && typeof fe.markComponentLayoutEffectUnmountStarted == "function" && fe.markComponentLayoutEffectUnmountStarted(
                t
              ), (e & qa) !== yu && (zh = !0), o = t, he(
                o,
                ZS,
                o,
                a,
                h
              ), (e & qa) !== yu && (zh = !1), (e & jl) !== yu ? fe !== null && typeof fe.markComponentPassiveEffectUnmountStopped == "function" && fe.markComponentPassiveEffectUnmountStopped() : (e & aa) !== yu && fe !== null && typeof fe.markComponentLayoutEffectUnmountStopped == "function" && fe.markComponentLayoutEffectUnmountStopped());
            }
            i = i.next;
          } while (i !== f);
        }
      } catch (v) {
        Me(t, t.return, v);
      }
    }
    function _y(e, t) {
      bn(e) ? (hn(), vc(t, e), La()) : vc(t, e);
    }
    function or(e, t, a) {
      bn(e) ? (hn(), gc(
        a,
        e,
        t
      ), La()) : gc(
        a,
        e,
        t
      );
    }
    function Uy(e) {
      var t = e.updateQueue;
      if (t !== null) {
        var a = e.stateNode;
        e.type.defaultProps || "ref" in e.memoizedProps || Sh || (a.props !== e.memoizedProps && console.error(
          "Expected %s props to match memoized props before processing the update queue. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
          de(e) || "instance"
        ), a.state !== e.memoizedState && console.error(
          "Expected %s state to match memoized state before processing the update queue. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
          de(e) || "instance"
        ));
        try {
          he(
            e,
            kp,
            t,
            a
          );
        } catch (i) {
          Me(e, e.return, i);
        }
      }
    }
    function tv(e, t, a) {
      return e.getSnapshotBeforeUpdate(t, a);
    }
    function Ug(e, t) {
      var a = t.memoizedProps, i = t.memoizedState;
      t = e.stateNode, e.type.defaultProps || "ref" in e.memoizedProps || Sh || (t.props !== e.memoizedProps && console.error(
        "Expected %s props to match memoized props before getSnapshotBeforeUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
        de(e) || "instance"
      ), t.state !== e.memoizedState && console.error(
        "Expected %s state to match memoized state before getSnapshotBeforeUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
        de(e) || "instance"
      ));
      try {
        var o = Si(
          e.type,
          a,
          e.elementType === e.type
        ), f = he(
          e,
          tv,
          t,
          o,
          i
        );
        a = nb, f !== void 0 || a.has(e.type) || (a.add(e.type), he(e, function() {
          console.error(
            "%s.getSnapshotBeforeUpdate(): A snapshot value (or null) must be returned. You have returned undefined.",
            de(e)
          );
        })), t.__reactInternalSnapshotBeforeUpdate = f;
      } catch (d) {
        Me(e, e.return, d);
      }
    }
    function cd(e, t, a) {
      a.props = Si(
        e.type,
        e.memoizedProps
      ), a.state = e.memoizedState, bn(e) ? (hn(), he(
        e,
        C1,
        e,
        t,
        a
      ), La()) : he(
        e,
        C1,
        e,
        t,
        a
      );
    }
    function lv(e) {
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
          if (bn(e))
            try {
              hn(), e.refCleanup = t(a);
            } finally {
              La();
            }
          else e.refCleanup = t(a);
        else
          typeof t == "string" ? console.error("String refs are no longer supported.") : t.hasOwnProperty("current") || console.error(
            "Unexpected ref object provided for %s. Use either a ref-setter function or React.createRef().",
            de(e)
          ), t.current = a;
      }
    }
    function _o(e, t) {
      try {
        he(e, lv, e);
      } catch (a) {
        Me(e, t, a);
      }
    }
    function $a(e, t) {
      var a = e.ref, i = e.refCleanup;
      if (a !== null)
        if (typeof i == "function")
          try {
            if (bn(e))
              try {
                hn(), he(e, i);
              } finally {
                La(e);
              }
            else he(e, i);
          } catch (o) {
            Me(e, t, o);
          } finally {
            e.refCleanup = null, e = e.alternate, e != null && (e.refCleanup = null);
          }
        else if (typeof a == "function")
          try {
            if (bn(e))
              try {
                hn(), he(e, a, null);
              } finally {
                La(e);
              }
            else he(e, a, null);
          } catch (o) {
            Me(e, t, o);
          }
        else a.current = null;
    }
    function Cy(e, t, a, i) {
      var o = e.memoizedProps, f = o.id, d = o.onCommit;
      o = o.onRender, t = t === null ? "mount" : "update", Gv && (t = "nested-update"), typeof o == "function" && o(
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
    function av(e, t, a, i) {
      var o = e.memoizedProps;
      e = o.id, o = o.onPostCommit, t = t === null ? "mount" : "update", Gv && (t = "nested-update"), typeof o == "function" && o(
        e,
        t,
        i,
        a
      );
    }
    function nv(e) {
      var t = e.type, a = e.memoizedProps, i = e.stateNode;
      try {
        he(
          e,
          Yu,
          i,
          t,
          a,
          e
        );
      } catch (o) {
        Me(e, e.return, o);
      }
    }
    function xy(e, t, a) {
      try {
        he(
          e,
          Wt,
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
    function Hy(e) {
      return e.tag === 5 || e.tag === 3 || e.tag === 26 || e.tag === 27 && lu(e.type) || e.tag === 4;
    }
    function bc(e) {
      e: for (; ; ) {
        for (; e.sibling === null; ) {
          if (e.return === null || Hy(e.return)) return null;
          e = e.return;
        }
        for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
          if (e.tag === 27 && lu(e.type) || e.flags & 2 || e.child === null || e.tag === 4) continue e;
          e.child.return = e, e = e.child;
        }
        if (!(e.flags & 2)) return e.stateNode;
      }
    }
    function fr(e, t, a) {
      var i = e.tag;
      if (i === 5 || i === 6)
        e = e.stateNode, t ? (a.nodeType === 9 ? a.body : a.nodeName === "HTML" ? a.ownerDocument.body : a).insertBefore(e, t) : (t = a.nodeType === 9 ? a.body : a.nodeName === "HTML" ? a.ownerDocument.body : a, t.appendChild(e), a = a._reactRootContainer, a != null || t.onclick !== null || (t.onclick = Bu));
      else if (i !== 4 && (i === 27 && lu(e.type) && (a = e.stateNode, t = null), e = e.child, e !== null))
        for (fr(e, t, a), e = e.sibling; e !== null; )
          fr(e, t, a), e = e.sibling;
    }
    function Sc(e, t, a) {
      var i = e.tag;
      if (i === 5 || i === 6)
        e = e.stateNode, t ? a.insertBefore(e, t) : a.appendChild(e);
      else if (i !== 4 && (i === 27 && lu(e.type) && (a = e.stateNode), e = e.child, e !== null))
        for (Sc(e, t, a), e = e.sibling; e !== null; )
          Sc(e, t, a), e = e.sibling;
    }
    function uv(e) {
      for (var t, a = e.return; a !== null; ) {
        if (Hy(a)) {
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
          t = t.stateNode, a = bc(e), Sc(
            e,
            a,
            t
          );
          break;
        case 5:
          a = t.stateNode, t.flags & 32 && (ju(a), t.flags &= -33), t = bc(e), Sc(
            e,
            t,
            a
          );
          break;
        case 3:
        case 4:
          t = t.stateNode.containerInfo, a = bc(e), fr(
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
    function Ny(e) {
      var t = e.stateNode, a = e.memoizedProps;
      try {
        he(
          e,
          Ca,
          e.type,
          a,
          t,
          e
        );
      } catch (i) {
        Me(e, e.return, i);
      }
    }
    function od(e, t) {
      if (e = e.containerInfo, N0 = yg, e = jp(e), Kh(e)) {
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
              var d = 0, h = -1, v = -1, b = 0, q = 0, L = e, x = null;
              t: for (; ; ) {
                for (var V; L !== a || o !== 0 && L.nodeType !== 3 || (h = d + o), L !== f || i !== 0 && L.nodeType !== 3 || (v = d + i), L.nodeType === 3 && (d += L.nodeValue.length), (V = L.firstChild) !== null; )
                  x = L, L = V;
                for (; ; ) {
                  if (L === e) break t;
                  if (x === a && ++b === o && (h = d), x === f && ++q === i && (v = d), (V = L.nextSibling) !== null) break;
                  L = x, x = L.parentNode;
                }
                L = V;
              }
              a = h === -1 || v === -1 ? null : { start: h, end: v };
            } else a = null;
          }
        a = a || { start: 0, end: 0 };
      } else a = null;
      for (w0 = {
        focusedElem: e,
        selectionRange: a
      }, yg = !1, kl = t; kl !== null; )
        if (t = kl, e = t.child, (t.subtreeFlags & 1024) !== 0 && e !== null)
          e.return = t, kl = e;
        else
          for (; kl !== null; ) {
            switch (e = t = kl, a = e.alternate, o = e.flags, e.tag) {
              case 0:
                break;
              case 11:
              case 15:
                break;
              case 1:
                (o & 1024) !== 0 && a !== null && Ug(e, a);
                break;
              case 3:
                if ((o & 1024) !== 0) {
                  if (e = e.stateNode.containerInfo, a = e.nodeType, a === 9)
                    Go(e);
                  else if (a === 1)
                    switch (e.nodeName) {
                      case "HEAD":
                      case "HTML":
                      case "BODY":
                        Go(e);
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
              e.return = t.return, kl = e;
              break;
            }
            kl = t.return;
          }
    }
    function wy(e, t, a) {
      var i = a.flags;
      switch (a.tag) {
        case 0:
        case 11:
        case 15:
          Jn(e, a), i & 4 && My(a, aa | mu);
          break;
        case 1:
          if (Jn(e, a), i & 4)
            if (e = a.stateNode, t === null)
              a.type.defaultProps || "ref" in a.memoizedProps || Sh || (e.props !== a.memoizedProps && console.error(
                "Expected %s props to match memoized props before componentDidMount. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
                de(a) || "instance"
              ), e.state !== a.memoizedState && console.error(
                "Expected %s state to match memoized state before componentDidMount. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
                de(a) || "instance"
              )), bn(a) ? (hn(), he(
                a,
                o0,
                a,
                e
              ), La()) : he(
                a,
                o0,
                a,
                e
              );
            else {
              var o = Si(
                a.type,
                t.memoizedProps
              );
              t = t.memoizedState, a.type.defaultProps || "ref" in a.memoizedProps || Sh || (e.props !== a.memoizedProps && console.error(
                "Expected %s props to match memoized props before componentDidUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
                de(a) || "instance"
              ), e.state !== a.memoizedState && console.error(
                "Expected %s state to match memoized state before componentDidUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
                de(a) || "instance"
              )), bn(a) ? (hn(), he(
                a,
                M1,
                a,
                e,
                o,
                t,
                e.__reactInternalSnapshotBeforeUpdate
              ), La()) : he(
                a,
                M1,
                a,
                e,
                o,
                t,
                e.__reactInternalSnapshotBeforeUpdate
              );
            }
          i & 64 && Uy(a), i & 512 && _o(a, a.return);
          break;
        case 3:
          if (t = dn(), Jn(e, a), i & 64 && (i = a.updateQueue, i !== null)) {
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
              he(
                a,
                kp,
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
          t === null && i & 4 && Ny(a);
        case 26:
        case 5:
          Jn(e, a), t === null && i & 4 && nv(a), i & 512 && _o(a, a.return);
          break;
        case 12:
          if (i & 4) {
            i = dn(), Jn(e, a), e = a.stateNode, e.effectDuration += fc(i);
            try {
              he(
                a,
                Cy,
                a,
                t,
                jv,
                e.effectDuration
              );
            } catch (d) {
              Me(a, a.return, d);
            }
          } else Jn(e, a);
          break;
        case 13:
          Jn(e, a), i & 4 && Uo(e, a), i & 64 && (e = a.memoizedState, e !== null && (e = e.dehydrated, e !== null && (a = vr.bind(
            null,
            a
          ), Lo(e, a))));
          break;
        case 22:
          if (i = a.memoizedState !== null || Kc, !i) {
            t = t !== null && t.memoizedState !== null || yl, o = Kc;
            var f = yl;
            Kc = i, (yl = t) && !f ? kn(
              e,
              a,
              (a.subtreeFlags & 8772) !== 0
            ) : Jn(e, a), Kc = o, yl = f;
          }
          break;
        case 30:
          break;
        default:
          Jn(e, a);
      }
    }
    function qy(e) {
      var t = e.alternate;
      t !== null && (e.alternate = null, qy(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && un(t)), e.stateNode = null, e._debugOwner = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
    }
    function Nu(e, t, a) {
      for (a = a.child; a !== null; )
        Tc(
          e,
          t,
          a
        ), a = a.sibling;
    }
    function Tc(e, t, a) {
      if (ql && typeof ql.onCommitFiberUnmount == "function")
        try {
          ql.onCommitFiberUnmount(Ni, a);
        } catch (f) {
          ga || (ga = !0, console.error(
            "React instrumentation encountered an error: %s",
            f
          ));
        }
      switch (a.tag) {
        case 26:
          yl || $a(a, t), Nu(
            e,
            t,
            a
          ), a.memoizedState ? a.memoizedState.count-- : a.stateNode && (a = a.stateNode, a.parentNode.removeChild(a));
          break;
        case 27:
          yl || $a(a, t);
          var i = Ol, o = an;
          lu(a.type) && (Ol = a.stateNode, an = !1), Nu(
            e,
            t,
            a
          ), he(
            a,
            Xo,
            a.stateNode
          ), Ol = i, an = o;
          break;
        case 5:
          yl || $a(a, t);
        case 6:
          if (i = Ol, o = an, Ol = null, Nu(
            e,
            t,
            a
          ), Ol = i, an = o, Ol !== null)
            if (an)
              try {
                he(
                  a,
                  Yo,
                  Ol,
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
                he(
                  a,
                  Ia,
                  Ol,
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
          Ol !== null && (an ? (e = Ol, jo(
            e.nodeType === 9 ? e.body : e.nodeName === "HTML" ? e.ownerDocument.body : e,
            a.stateNode
          ), Nc(e)) : jo(Ol, a.stateNode));
          break;
        case 4:
          i = Ol, o = an, Ol = a.stateNode.containerInfo, an = !0, Nu(
            e,
            t,
            a
          ), Ol = i, an = o;
          break;
        case 0:
        case 11:
        case 14:
        case 15:
          yl || gc(
            qa,
            a,
            t
          ), yl || id(
            a,
            t,
            aa
          ), Nu(
            e,
            t,
            a
          );
          break;
        case 1:
          yl || ($a(a, t), i = a.stateNode, typeof i.componentWillUnmount == "function" && cd(
            a,
            t,
            i
          )), Nu(
            e,
            t,
            a
          );
          break;
        case 21:
          Nu(
            e,
            t,
            a
          );
          break;
        case 22:
          yl = (i = yl) || a.memoizedState !== null, Nu(
            e,
            t,
            a
          ), yl = i;
          break;
        default:
          Nu(
            e,
            t,
            a
          );
      }
    }
    function Uo(e, t) {
      if (t.memoizedState === null && (e = t.alternate, e !== null && (e = e.memoizedState, e !== null && (e = e.dehydrated, e !== null))))
        try {
          he(
            t,
            Ua,
            e
          );
        } catch (a) {
          Me(t, t.return, a);
        }
    }
    function fd(e) {
      switch (e.tag) {
        case 13:
        case 19:
          var t = e.stateNode;
          return t === null && (t = e.stateNode = new ub()), t;
        case 22:
          return e = e.stateNode, t = e._retryCache, t === null && (t = e._retryCache = new ub()), t;
        default:
          throw Error(
            "Unexpected Suspense handler tag (" + e.tag + "). This is a bug in React."
          );
      }
    }
    function Ec(e, t) {
      var a = fd(e);
      t.forEach(function(i) {
        var o = Oi.bind(null, e, i);
        if (!a.has(i)) {
          if (a.add(i), It)
            if (Th !== null && Eh !== null)
              qo(Eh, Th);
            else
              throw Error(
                "Expected finished root and lanes to be set. This is a bug in React."
              );
          i.then(o, o);
        }
      });
    }
    function Xl(e, t) {
      var a = t.deletions;
      if (a !== null)
        for (var i = 0; i < a.length; i++) {
          var o = e, f = t, d = a[i], h = f;
          e: for (; h !== null; ) {
            switch (h.tag) {
              case 27:
                if (lu(h.type)) {
                  Ol = h.stateNode, an = !1;
                  break e;
                }
                break;
              case 5:
                Ol = h.stateNode, an = !1;
                break e;
              case 3:
              case 4:
                Ol = h.stateNode.containerInfo, an = !0;
                break e;
            }
            h = h.return;
          }
          if (Ol === null)
            throw Error(
              "Expected to find a host parent. This error is likely caused by a bug in React. Please file an issue."
            );
          Tc(o, f, d), Ol = null, an = !1, o = d, f = o.alternate, f !== null && (f.return = null), o.return = null;
        }
      if (t.subtreeFlags & 13878)
        for (t = t.child; t !== null; )
          By(t, e), t = t.sibling;
    }
    function By(e, t) {
      var a = e.alternate, i = e.flags;
      switch (e.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          Xl(t, e), ha(e), i & 4 && (gc(
            qa | mu,
            e,
            e.return
          ), vc(qa | mu, e), id(
            e,
            e.return,
            aa | mu
          ));
          break;
        case 1:
          Xl(t, e), ha(e), i & 512 && (yl || a === null || $a(a, a.return)), i & 64 && Kc && (e = e.updateQueue, e !== null && (i = e.callbacks, i !== null && (a = e.shared.hiddenCallbacks, e.shared.hiddenCallbacks = a === null ? i : a.concat(i))));
          break;
        case 26:
          var o = Fu;
          if (Xl(t, e), ha(e), i & 512 && (yl || a === null || $a(a, a.return)), i & 4)
            if (t = a !== null ? a.memoizedState : null, i = e.memoizedState, a === null)
              if (i === null)
                if (e.stateNode === null) {
                  e: {
                    i = e.type, a = e.memoizedProps, t = o.ownerDocument || o;
                    t: switch (i) {
                      case "title":
                        o = t.getElementsByTagName("title")[0], (!o || o[ef] || o[Kl] || o.namespaceURI === nf || o.hasAttribute("itemprop")) && (o = t.createElement(i), t.head.insertBefore(
                          o,
                          t.querySelector("head > title")
                        )), $t(o, i, a), o[Kl] = e, z(o), i = o;
                        break e;
                      case "link":
                        var f = mm(
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
                        o = t.createElement(i), $t(o, i, a), t.head.appendChild(o);
                        break;
                      case "meta":
                        if (f = mm(
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
                        o = t.createElement(i), $t(o, i, a), t.head.appendChild(o);
                        break;
                      default:
                        throw Error(
                          'getNodesForType encountered a type it did not expect: "' + i + '". This is a bug in React.'
                        );
                    }
                    o[Kl] = e, z(o), i = o;
                  }
                  e.stateNode = i;
                } else
                  pm(
                    o,
                    e.type,
                    e.stateNode
                  );
              else
                e.stateNode = xd(
                  o,
                  i,
                  e.memoizedProps
                );
            else
              t !== i ? (t === null ? a.stateNode !== null && (a = a.stateNode, a.parentNode.removeChild(a)) : t.count--, i === null ? pm(
                o,
                e.type,
                e.stateNode
              ) : xd(
                o,
                i,
                e.memoizedProps
              )) : i === null && e.stateNode !== null && xy(
                e,
                e.memoizedProps,
                a.memoizedProps
              );
          break;
        case 27:
          Xl(t, e), ha(e), i & 512 && (yl || a === null || $a(a, a.return)), a !== null && i & 4 && xy(
            e,
            e.memoizedProps,
            a.memoizedProps
          );
          break;
        case 5:
          if (Xl(t, e), ha(e), i & 512 && (yl || a === null || $a(a, a.return)), e.flags & 32) {
            t = e.stateNode;
            try {
              he(e, ju, t);
            } catch (q) {
              Me(e, e.return, q);
            }
          }
          i & 4 && e.stateNode != null && (t = e.memoizedProps, xy(
            e,
            t,
            a !== null ? a.memoizedProps : t
          )), i & 1024 && (m0 = !0, e.type !== "form" && console.error(
            "Unexpected host component type. Expected a form. This is a bug in React."
          ));
          break;
        case 6:
          if (Xl(t, e), ha(e), i & 4) {
            if (e.stateNode === null)
              throw Error(
                "This should have a text node initialized. This error is likely caused by a bug in React. Please file an issue."
              );
            i = e.memoizedProps, a = a !== null ? a.memoizedProps : i, t = e.stateNode;
            try {
              he(
                e,
                Uc,
                t,
                a,
                i
              );
            } catch (q) {
              Me(e, e.return, q);
            }
          }
          break;
        case 3:
          if (o = dn(), sg = null, f = Fu, Fu = br(t.containerInfo), Xl(t, e), Fu = f, ha(e), i & 4 && a !== null && a.memoizedState.isDehydrated)
            try {
              he(
                e,
                rm,
                t.containerInfo
              );
            } catch (q) {
              Me(e, e.return, q);
            }
          m0 && (m0 = !1, Rc(e)), t.effectDuration += si(o);
          break;
        case 4:
          i = Fu, Fu = br(
            e.stateNode.containerInfo
          ), Xl(t, e), ha(e), Fu = i;
          break;
        case 12:
          i = dn(), Xl(t, e), ha(e), e.stateNode.effectDuration += fc(i);
          break;
        case 13:
          Xl(t, e), ha(e), e.child.flags & 8192 && e.memoizedState !== null != (a !== null && a.memoizedState !== null) && (T0 = iu()), i & 4 && (i = e.updateQueue, i !== null && (e.updateQueue = null, Ec(e, i)));
          break;
        case 22:
          o = e.memoizedState !== null;
          var h = a !== null && a.memoizedState !== null, v = Kc, b = yl;
          if (Kc = v || o, yl = b || h, Xl(t, e), yl = b, Kc = v, ha(e), i & 8192)
            e: for (t = e.stateNode, t._visibility = o ? t._visibility & ~wv : t._visibility | wv, o && (a === null || h || Kc || yl || Ql(e)), a = null, t = e; ; ) {
              if (t.tag === 5 || t.tag === 26) {
                if (a === null) {
                  h = a = t;
                  try {
                    f = h.stateNode, o ? he(h, pa, f) : he(
                      h,
                      om,
                      h.stateNode,
                      h.memoizedProps
                    );
                  } catch (q) {
                    Me(h, h.return, q);
                  }
                }
              } else if (t.tag === 6) {
                if (a === null) {
                  h = t;
                  try {
                    d = h.stateNode, o ? he(h, cm, d) : he(
                      h,
                      _d,
                      d,
                      h.memoizedProps
                    );
                  } catch (q) {
                    Me(h, h.return, q);
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
          i & 4 && (i = e.updateQueue, i !== null && (a = i.retryQueue, a !== null && (i.retryQueue = null, Ec(e, a))));
          break;
        case 19:
          Xl(t, e), ha(e), i & 4 && (i = e.updateQueue, i !== null && (e.updateQueue = null, Ec(e, i)));
          break;
        case 30:
          break;
        case 21:
          break;
        default:
          Xl(t, e), ha(e);
      }
    }
    function ha(e) {
      var t = e.flags;
      if (t & 2) {
        try {
          he(e, uv, e);
        } catch (a) {
          Me(e, e.return, a);
        }
        e.flags &= -3;
      }
      t & 4096 && (e.flags &= -4097);
    }
    function Rc(e) {
      if (e.subtreeFlags & 1024)
        for (e = e.child; e !== null; ) {
          var t = e;
          Rc(t), t.tag === 5 && t.flags & 1024 && t.stateNode.reset(), e = e.sibling;
        }
    }
    function Jn(e, t) {
      if (t.subtreeFlags & 8772)
        for (t = t.child; t !== null; )
          wy(e, t.alternate, t), t = t.sibling;
    }
    function _a(e) {
      switch (e.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          id(
            e,
            e.return,
            aa
          ), Ql(e);
          break;
        case 1:
          $a(e, e.return);
          var t = e.stateNode;
          typeof t.componentWillUnmount == "function" && cd(
            e,
            e.return,
            t
          ), Ql(e);
          break;
        case 27:
          he(
            e,
            Xo,
            e.stateNode
          );
        case 26:
        case 5:
          $a(e, e.return), Ql(e);
          break;
        case 22:
          e.memoizedState === null && Ql(e);
          break;
        case 30:
          Ql(e);
          break;
        default:
          Ql(e);
      }
    }
    function Ql(e) {
      for (e = e.child; e !== null; )
        _a(e), e = e.sibling;
    }
    function wu(e, t, a, i) {
      var o = a.flags;
      switch (a.tag) {
        case 0:
        case 11:
        case 15:
          kn(
            e,
            a,
            i
          ), My(a, aa);
          break;
        case 1:
          if (kn(
            e,
            a,
            i
          ), t = a.stateNode, typeof t.componentDidMount == "function" && he(
            a,
            o0,
            a,
            t
          ), t = a.updateQueue, t !== null) {
            e = a.stateNode;
            try {
              he(
                a,
                po,
                t,
                e
              );
            } catch (f) {
              Me(a, a.return, f);
            }
          }
          i && o & 64 && Uy(a), _o(a, a.return);
          break;
        case 27:
          Ny(a);
        case 26:
        case 5:
          kn(
            e,
            a,
            i
          ), i && t === null && o & 4 && nv(a), _o(a, a.return);
          break;
        case 12:
          if (i && o & 4) {
            o = dn(), kn(
              e,
              a,
              i
            ), i = a.stateNode, i.effectDuration += fc(o);
            try {
              he(
                a,
                Cy,
                a,
                t,
                jv,
                i.effectDuration
              );
            } catch (f) {
              Me(a, a.return, f);
            }
          } else
            kn(
              e,
              a,
              i
            );
          break;
        case 13:
          kn(
            e,
            a,
            i
          ), i && o & 4 && Uo(e, a);
          break;
        case 22:
          a.memoizedState === null && kn(
            e,
            a,
            i
          ), _o(a, a.return);
          break;
        case 30:
          break;
        default:
          kn(
            e,
            a,
            i
          );
      }
    }
    function kn(e, t, a) {
      for (a = a && (t.subtreeFlags & 8772) !== 0, t = t.child; t !== null; )
        wu(
          e,
          t.alternate,
          t,
          a
        ), t = t.sibling;
    }
    function $n(e, t) {
      var a = null;
      e !== null && e.memoizedState !== null && e.memoizedState.cachePool !== null && (a = e.memoizedState.cachePool.pool), e = null, t.memoizedState !== null && t.memoizedState.cachePool !== null && (e = t.memoizedState.cachePool.pool), e !== a && (e != null && oc(e), a != null && wn(a));
    }
    function Sn(e, t) {
      e = null, t.alternate !== null && (e = t.alternate.memoizedState.cache), t = t.memoizedState.cache, t !== e && (oc(t), e != null && wn(e));
    }
    function Mt(e, t, a, i) {
      if (t.subtreeFlags & 10256)
        for (t = t.child; t !== null; )
          rr(
            e,
            t,
            a,
            i
          ), t = t.sibling;
    }
    function rr(e, t, a, i) {
      var o = t.flags;
      switch (t.tag) {
        case 0:
        case 11:
        case 15:
          Mt(
            e,
            t,
            a,
            i
          ), o & 2048 && _y(t, jl | mu);
          break;
        case 1:
          Mt(
            e,
            t,
            a,
            i
          );
          break;
        case 3:
          var f = dn();
          Mt(
            e,
            t,
            a,
            i
          ), o & 2048 && (a = null, t.alternate !== null && (a = t.alternate.memoizedState.cache), t = t.memoizedState.cache, t !== a && (oc(t), a != null && wn(a))), e.passiveEffectDuration += si(f);
          break;
        case 12:
          if (o & 2048) {
            o = dn(), Mt(
              e,
              t,
              a,
              i
            ), e = t.stateNode, e.passiveEffectDuration += fc(o);
            try {
              he(
                t,
                av,
                t,
                t.alternate,
                jv,
                e.passiveEffectDuration
              );
            } catch (h) {
              Me(t, t.return, h);
            }
          } else
            Mt(
              e,
              t,
              a,
              i
            );
          break;
        case 13:
          Mt(
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
          t.memoizedState !== null ? f._visibility & Gc ? Mt(
            e,
            t,
            a,
            i
          ) : Co(
            e,
            t
          ) : f._visibility & Gc ? Mt(
            e,
            t,
            a,
            i
          ) : (f._visibility |= Gc, Ei(
            e,
            t,
            a,
            i,
            (t.subtreeFlags & 10256) !== 0
          )), o & 2048 && $n(d, t);
          break;
        case 24:
          Mt(
            e,
            t,
            a,
            i
          ), o & 2048 && Sn(t.alternate, t);
          break;
        default:
          Mt(
            e,
            t,
            a,
            i
          );
      }
    }
    function Ei(e, t, a, i, o) {
      for (o = o && (t.subtreeFlags & 10256) !== 0, t = t.child; t !== null; )
        rd(
          e,
          t,
          a,
          i,
          o
        ), t = t.sibling;
    }
    function rd(e, t, a, i, o) {
      var f = t.flags;
      switch (t.tag) {
        case 0:
        case 11:
        case 15:
          Ei(
            e,
            t,
            a,
            i,
            o
          ), _y(t, jl);
          break;
        case 23:
          break;
        case 22:
          var d = t.stateNode;
          t.memoizedState !== null ? d._visibility & Gc ? Ei(
            e,
            t,
            a,
            i,
            o
          ) : Co(
            e,
            t
          ) : (d._visibility |= Gc, Ei(
            e,
            t,
            a,
            i,
            o
          )), o && f & 2048 && $n(
            t.alternate,
            t
          );
          break;
        case 24:
          Ei(
            e,
            t,
            a,
            i,
            o
          ), o && f & 2048 && Sn(t.alternate, t);
          break;
        default:
          Ei(
            e,
            t,
            a,
            i,
            o
          );
      }
    }
    function Co(e, t) {
      if (t.subtreeFlags & 10256)
        for (t = t.child; t !== null; ) {
          var a = e, i = t, o = i.flags;
          switch (i.tag) {
            case 22:
              Co(
                a,
                i
              ), o & 2048 && $n(
                i.alternate,
                i
              );
              break;
            case 24:
              Co(
                a,
                i
              ), o & 2048 && Sn(
                i.alternate,
                i
              );
              break;
            default:
              Co(
                a,
                i
              );
          }
          t = t.sibling;
        }
    }
    function Ac(e) {
      if (e.subtreeFlags & up)
        for (e = e.child; e !== null; )
          Ri(e), e = e.sibling;
    }
    function Ri(e) {
      switch (e.tag) {
        case 26:
          Ac(e), e.flags & up && e.memoizedState !== null && pv(
            Fu,
            e.memoizedState,
            e.memoizedProps
          );
          break;
        case 5:
          Ac(e);
          break;
        case 3:
        case 4:
          var t = Fu;
          Fu = br(
            e.stateNode.containerInfo
          ), Ac(e), Fu = t;
          break;
        case 22:
          e.memoizedState === null && (t = e.alternate, t !== null && t.memoizedState !== null ? (t = up, up = 16777216, Ac(e), up = t) : Ac(e));
          break;
        default:
          Ac(e);
      }
    }
    function sr(e) {
      var t = e.alternate;
      if (t !== null && (e = t.child, e !== null)) {
        t.child = null;
        do
          t = e.sibling, e.sibling = null, e = t;
        while (e !== null);
      }
    }
    function xo(e) {
      var t = e.deletions;
      if ((e.flags & 16) !== 0) {
        if (t !== null)
          for (var a = 0; a < t.length; a++) {
            var i = t[a];
            kl = i, jy(
              i,
              e
            );
          }
        sr(e);
      }
      if (e.subtreeFlags & 10256)
        for (e = e.child; e !== null; )
          Yy(e), e = e.sibling;
    }
    function Yy(e) {
      switch (e.tag) {
        case 0:
        case 11:
        case 15:
          xo(e), e.flags & 2048 && or(
            e,
            e.return,
            jl | mu
          );
          break;
        case 3:
          var t = dn();
          xo(e), e.stateNode.passiveEffectDuration += si(t);
          break;
        case 12:
          t = dn(), xo(e), e.stateNode.passiveEffectDuration += fc(t);
          break;
        case 22:
          t = e.stateNode, e.memoizedState !== null && t._visibility & Gc && (e.return === null || e.return.tag !== 13) ? (t._visibility &= ~Gc, dr(e)) : xo(e);
          break;
        default:
          xo(e);
      }
    }
    function dr(e) {
      var t = e.deletions;
      if ((e.flags & 16) !== 0) {
        if (t !== null)
          for (var a = 0; a < t.length; a++) {
            var i = t[a];
            kl = i, jy(
              i,
              e
            );
          }
        sr(e);
      }
      for (e = e.child; e !== null; )
        hr(e), e = e.sibling;
    }
    function hr(e) {
      switch (e.tag) {
        case 0:
        case 11:
        case 15:
          or(
            e,
            e.return,
            jl
          ), dr(e);
          break;
        case 22:
          var t = e.stateNode;
          t._visibility & Gc && (t._visibility &= ~Gc, dr(e));
          break;
        default:
          dr(e);
      }
    }
    function jy(e, t) {
      for (; kl !== null; ) {
        var a = kl, i = a;
        switch (i.tag) {
          case 0:
          case 11:
          case 15:
            or(
              i,
              t,
              jl
            );
            break;
          case 23:
          case 22:
            i.memoizedState !== null && i.memoizedState.cachePool !== null && (i = i.memoizedState.cachePool.pool, i != null && oc(i));
            break;
          case 24:
            wn(i.memoizedState.cache);
        }
        if (i = a.child, i !== null) i.return = a, kl = i;
        else
          e: for (a = e; kl !== null; ) {
            i = kl;
            var o = i.sibling, f = i.return;
            if (qy(i), i === a) {
              kl = null;
              break e;
            }
            if (o !== null) {
              o.return = f, kl = o;
              break e;
            }
            kl = f;
          }
      }
    }
    function Gy() {
      JS.forEach(function(e) {
        return e();
      });
    }
    function Ly() {
      var e = typeof IS_REACT_ACT_ENVIRONMENT < "u" ? IS_REACT_ACT_ENVIRONMENT : void 0;
      return e || Y.actQueue === null || console.error(
        "The current testing environment is not configured to support act(...)"
      ), e;
    }
    function ya(e) {
      if ((Et & Ba) !== An && nt !== 0)
        return nt & -nt;
      var t = Y.T;
      return t !== null ? (t._updatedFibers || (t._updatedFibers = /* @__PURE__ */ new Set()), t._updatedFibers.add(e), e = Zr, e !== 0 ? e : Wy()) : Ef();
    }
    function iv() {
      Dn === 0 && (Dn = (nt & 536870912) === 0 || mt ? Je() : 536870912);
      var e = vu.current;
      return e !== null && (e.flags |= 32), Dn;
    }
    function Jt(e, t, a) {
      if (zh && console.error("useInsertionEffect must not schedule updates."), D0 && (Pv = !0), (e === wt && (_t === Wr || _t === Fr) || e.cancelPendingCommit !== null) && (Dc(e, 0), qu(
        e,
        nt,
        Dn,
        !1
      )), Su(e, a), (Et & Ba) !== 0 && e === wt) {
        if (Sa)
          switch (t.tag) {
            case 0:
            case 11:
            case 15:
              e = at && de(at) || "Unknown", mb.has(e) || (mb.add(e), t = de(t) || "Unknown", console.error(
                "Cannot update a component (`%s`) while rendering a different component (`%s`). To locate the bad setState() call inside `%s`, follow the stack trace as described in https://react.dev/link/setstate-in-render",
                t,
                e,
                e
              ));
              break;
            case 1:
              yb || (console.error(
                "Cannot update during an existing state transition (such as within `render`). Render methods should be a pure function of props and state."
              ), yb = !0);
          }
      } else
        It && ja(e, t, a), rv(t), e === wt && ((Et & Ba) === An && (df |= a), il === $r && qu(
          e,
          nt,
          Dn,
          !1
        )), Wa(e);
    }
    function dl(e, t, a) {
      if ((Et & (Ba | Iu)) !== An)
        throw Error("Should not already be working.");
      var i = !a && (t & 124) === 0 && (t & e.expiredLanes) === 0 || Pu(e, t), o = i ? Xy(e, t) : yd(e, t, !0), f = i;
      do {
        if (o === Jc) {
          Oh && !i && qu(e, t, 0, !1);
          break;
        } else {
          if (a = e.current.alternate, f && !cv(a)) {
            o = yd(e, t, !1), f = !1;
            continue;
          }
          if (o === Rh) {
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
                if (v && (Dc(
                  o,
                  h
                ).flags |= 256), h = yd(
                  o,
                  h,
                  !1
                ), h !== Rh) {
                  if (b0 && !v) {
                    o.errorRecoveryDisabledLanes |= f, df |= f, o = $r;
                    break e;
                  }
                  o = Ya, Ya = d, o !== null && (Ya === null ? Ya = o : Ya.push.apply(
                    Ya,
                    o
                  ));
                }
                o = h;
              }
              if (f = !1, o !== Rh) continue;
            }
          }
          if (o === cp) {
            Dc(e, 0), qu(e, t, 0, !0);
            break;
          }
          e: {
            switch (i = e, o) {
              case Jc:
              case cp:
                throw Error("Root did not complete. This is a bug in React.");
              case $r:
                if ((t & 4194048) !== t) break;
              case Wv:
                qu(
                  i,
                  t,
                  Dn,
                  !rf
                );
                break e;
              case Rh:
                Ya = null;
                break;
              case p0:
              case ib:
                break;
              default:
                throw Error("Unknown root exit status.");
            }
            if (Y.actQueue !== null)
              bd(
                i,
                a,
                t,
                Ya,
                dp,
                Fv,
                Dn,
                df,
                Ir
              );
            else {
              if ((t & 62914560) === t && (f = T0 + ob - iu(), 10 < f)) {
                if (qu(
                  i,
                  t,
                  Dn,
                  !rf
                ), vl(i, 0, !0) !== 0) break e;
                i.timeoutHandle = Rb(
                  Tl.bind(
                    null,
                    i,
                    a,
                    Ya,
                    dp,
                    Fv,
                    t,
                    Dn,
                    df,
                    Ir,
                    rf,
                    o,
                    FS,
                    r1,
                    0
                  ),
                  f
                );
                break e;
              }
              Tl(
                i,
                a,
                Ya,
                dp,
                Fv,
                t,
                Dn,
                df,
                Ir,
                rf,
                o,
                $S,
                r1,
                0
              );
            }
          }
        }
        break;
      } while (!0);
      Wa(e);
    }
    function Tl(e, t, a, i, o, f, d, h, v, b, q, L, x, V) {
      if (e.timeoutHandle = as, L = t.subtreeFlags, (L & 8192 || (L & 16785408) === 16785408) && (gp = { stylesheets: null, count: 0, unsuspend: mv }, Ri(t), L = vv(), L !== null)) {
        e.cancelPendingCommit = L(
          bd.bind(
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
            q,
            WS,
            x,
            V
          )
        ), qu(
          e,
          f,
          d,
          !b
        );
        return;
      }
      bd(
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
    function cv(e) {
      for (var t = e; ; ) {
        var a = t.tag;
        if ((a === 0 || a === 11 || a === 15) && t.flags & 16384 && (a = t.updateQueue, a !== null && (a = a.stores, a !== null)))
          for (var i = 0; i < a.length; i++) {
            var o = a[i], f = o.getSnapshot;
            o = o.value;
            try {
              if (!Na(f(), o)) return !1;
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
    function qu(e, t, a, i) {
      t &= ~S0, t &= ~df, e.suspendedLanes |= t, e.pingedLanes &= ~t, i && (e.warmLanes |= t), i = e.expirationTimes;
      for (var o = t; 0 < o; ) {
        var f = 31 - Zl(o), d = 1 << f;
        i[f] = -1, o &= ~d;
      }
      a !== 0 && Tf(e, a, t);
    }
    function Oc() {
      return (Et & (Ba | Iu)) === An ? (zc(0), !1) : !0;
    }
    function sd() {
      if (at !== null) {
        if (_t === nn)
          var e = at.return;
        else
          e = at, Ms(), pn(e), ph = null, ap = 0, e = at;
        for (; e !== null; )
          zy(e.alternate, e), e = e.return;
        at = null;
      }
    }
    function Dc(e, t) {
      var a = e.timeoutHandle;
      a !== as && (e.timeoutHandle = as, rT(a)), a = e.cancelPendingCommit, a !== null && (e.cancelPendingCommit = null, a()), sd(), wt = e, at = a = Hn(e.current, null), nt = t, _t = nn, On = null, rf = !1, Oh = Pu(e, t), b0 = !1, il = Jc, Ir = Dn = S0 = df = sf = 0, Ya = sp = null, Fv = !1, (t & 8) !== 0 && (t |= t & 32);
      var i = e.entangledLanes;
      if (i !== 0)
        for (e = e.entanglements, i &= t; 0 < i; ) {
          var o = 31 - Zl(i), f = 1 << o;
          t |= e[o], i &= ~f;
        }
      return Xi = t, Nf(), t = o1(), 1e3 < t - c1 && (Y.recentlyCreatedOwnerStacks = 0, c1 = t), $u.discardPendingWarnings(), a;
    }
    function yr(e, t) {
      Be = null, Y.H = kv, Y.getCurrentStack = null, Sa = !1, Ha = null, t === Pm || t === Xv ? (t = ry(), _t = fp) : t === h1 ? (t = ry(), _t = cb) : _t = t === F1 ? g0 : t !== null && typeof t == "object" && typeof t.then == "function" ? Ah : op, On = t;
      var a = at;
      if (a === null)
        il = cp, Mo(
          e,
          Oa(t, e.current)
        );
      else
        switch (a.mode & la && zu(a), ua(), _t) {
          case op:
            fe !== null && typeof fe.markComponentErrored == "function" && fe.markComponentErrored(
              a,
              t,
              nt
            );
            break;
          case Wr:
          case Fr:
          case fp:
          case Ah:
          case rp:
            fe !== null && typeof fe.markComponentSuspended == "function" && fe.markComponentSuspended(
              a,
              t,
              nt
            );
        }
    }
    function dd() {
      var e = Y.H;
      return Y.H = kv, e === null ? kv : e;
    }
    function Vy() {
      var e = Y.A;
      return Y.A = KS, e;
    }
    function hd() {
      il = $r, rf || (nt & 4194048) !== nt && vu.current !== null || (Oh = !0), (sf & 134217727) === 0 && (df & 134217727) === 0 || wt === null || qu(
        wt,
        nt,
        Dn,
        !1
      );
    }
    function yd(e, t, a) {
      var i = Et;
      Et |= Ba;
      var o = dd(), f = Vy();
      if (wt !== e || nt !== t) {
        if (It) {
          var d = e.memoizedUpdaters;
          0 < d.size && (qo(e, nt), d.clear()), zl(e, t);
        }
        dp = null, Dc(e, t);
      }
      Mn(t), t = !1, d = il;
      e: do
        try {
          if (_t !== nn && at !== null) {
            var h = at, v = On;
            switch (_t) {
              case g0:
                sd(), d = Wv;
                break e;
              case fp:
              case Wr:
              case Fr:
              case Ah:
                vu.current === null && (t = !0);
                var b = _t;
                if (_t = nn, On = null, Ai(e, h, v, b), a && Oh) {
                  d = Jc;
                  break e;
                }
                break;
              default:
                b = _t, _t = nn, On = null, Ai(e, h, v, b);
            }
          }
          md(), d = il;
          break;
        } catch (q) {
          yr(e, q);
        }
      while (!0);
      return t && e.shellSuspendCounter++, Ms(), Et = i, Y.H = o, Y.A = f, Ki(), at === null && (wt = null, nt = 0, Nf()), d;
    }
    function md() {
      for (; at !== null; ) Zy(at);
    }
    function Xy(e, t) {
      var a = Et;
      Et |= Ba;
      var i = dd(), o = Vy();
      if (wt !== e || nt !== t) {
        if (It) {
          var f = e.memoizedUpdaters;
          0 < f.size && (qo(e, nt), f.clear()), zl(e, t);
        }
        dp = null, Iv = iu() + fb, Dc(e, t);
      } else
        Oh = Pu(
          e,
          t
        );
      Mn(t);
      e: do
        try {
          if (_t !== nn && at !== null)
            t: switch (t = at, f = On, _t) {
              case op:
                _t = nn, On = null, Ai(
                  e,
                  t,
                  f,
                  op
                );
                break;
              case Wr:
              case Fr:
                if (fy(f)) {
                  _t = nn, On = null, pd(t);
                  break;
                }
                t = function() {
                  _t !== Wr && _t !== Fr || wt !== e || (_t = rp), Wa(e);
                }, f.then(t, t);
                break e;
              case fp:
                _t = rp;
                break e;
              case cb:
                _t = v0;
                break e;
              case rp:
                fy(f) ? (_t = nn, On = null, pd(t)) : (_t = nn, On = null, Ai(
                  e,
                  t,
                  f,
                  rp
                ));
                break;
              case v0:
                var d = null;
                switch (at.tag) {
                  case 26:
                    d = at.memoizedState;
                  case 5:
                  case 27:
                    var h = at;
                    if (!d || Sr(d)) {
                      _t = nn, On = null;
                      var v = h.sibling;
                      if (v !== null) at = v;
                      else {
                        var b = h.return;
                        b !== null ? (at = b, mr(b)) : at = null;
                      }
                      break t;
                    }
                    break;
                  default:
                    console.error(
                      "Unexpected type of fiber triggered a suspensey commit. This is a bug in React."
                    );
                }
                _t = nn, On = null, Ai(
                  e,
                  t,
                  f,
                  v0
                );
                break;
              case Ah:
                _t = nn, On = null, Ai(
                  e,
                  t,
                  f,
                  Ah
                );
                break;
              case g0:
                sd(), il = Wv;
                break e;
              default:
                throw Error(
                  "Unexpected SuspendedReason. This is a bug in React."
                );
            }
          Y.actQueue !== null ? md() : Qy();
          break;
        } catch (q) {
          yr(e, q);
        }
      while (!0);
      return Ms(), Y.H = i, Y.A = o, Et = a, at !== null ? (fe !== null && typeof fe.markRenderYielded == "function" && fe.markRenderYielded(), Jc) : (Ki(), wt = null, nt = 0, Nf(), il);
    }
    function Qy() {
      for (; at !== null && !Av(); )
        Zy(at);
    }
    function Zy(e) {
      var t = e.alternate;
      (e.mode & la) !== Gt ? (_s(e), t = he(
        e,
        ud,
        t,
        e,
        Xi
      ), zu(e)) : t = he(
        e,
        ud,
        t,
        e,
        Xi
      ), e.memoizedProps = e.pendingProps, t === null ? mr(e) : at = t;
    }
    function pd(e) {
      var t = he(e, vd, e);
      e.memoizedProps = e.pendingProps, t === null ? mr(e) : at = t;
    }
    function vd(e) {
      var t = e.alternate, a = (e.mode & la) !== Gt;
      switch (a && _s(e), e.tag) {
        case 15:
        case 0:
          t = Ey(
            t,
            e,
            e.pendingProps,
            e.type,
            void 0,
            nt
          );
          break;
        case 11:
          t = Ey(
            t,
            e,
            e.pendingProps,
            e.type.render,
            e.ref,
            nt
          );
          break;
        case 5:
          pn(e);
        default:
          zy(t, e), e = at = Wh(e, Xi), t = ud(t, e, Xi);
      }
      return a && zu(e), t;
    }
    function Ai(e, t, a, i) {
      Ms(), pn(t), ph = null, ap = 0;
      var o = t.return;
      try {
        if (tr(
          e,
          o,
          t,
          a,
          nt
        )) {
          il = cp, Mo(
            e,
            Oa(a, e.current)
          ), at = null;
          return;
        }
      } catch (f) {
        if (o !== null) throw at = o, f;
        il = cp, Mo(
          e,
          Oa(a, e.current)
        ), at = null;
        return;
      }
      t.flags & 32768 ? (mt || i === op ? e = !0 : Oh || (nt & 536870912) !== 0 ? e = !1 : (rf = e = !0, (i === Wr || i === Fr || i === fp || i === Ah) && (i = vu.current, i !== null && i.tag === 13 && (i.flags |= 16384))), gd(t, e)) : mr(t);
    }
    function mr(e) {
      var t = e;
      do {
        if ((t.flags & 32768) !== 0) {
          gd(
            t,
            rf
          );
          return;
        }
        var a = t.alternate;
        if (e = t.return, _s(t), a = he(
          t,
          Pp,
          a,
          t,
          Xi
        ), (t.mode & la) !== Gt && rc(t), a !== null) {
          at = a;
          return;
        }
        if (t = t.sibling, t !== null) {
          at = t;
          return;
        }
        at = t = e;
      } while (t !== null);
      il === Jc && (il = ib);
    }
    function gd(e, t) {
      do {
        var a = ev(e.alternate, e);
        if (a !== null) {
          a.flags &= 32767, at = a;
          return;
        }
        if ((e.mode & la) !== Gt) {
          rc(e), a = e.actualDuration;
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
      il = Wv, at = null;
    }
    function bd(e, t, a, i, o, f, d, h, v) {
      e.cancelPendingCommit = null;
      do
        Ho();
      while (na !== Pr);
      if ($u.flushLegacyContextWarning(), $u.flushPendingUnsafeLifecycleWarnings(), (Et & (Ba | Iu)) !== An)
        throw Error("Should not already be working.");
      if (fe !== null && typeof fe.markCommitStarted == "function" && fe.markCommitStarted(a), t === null) xe();
      else {
        if (a === 0 && console.error(
          "finishedLanes should not be empty during a commit. This is a bug in React."
        ), t === e.current)
          throw Error(
            "Cannot commit the same tree as before. This error is likely caused by a bug in React. Please file an issue."
          );
        if (f = t.lanes | t.childLanes, f |= $g, os(
          e,
          a,
          f,
          d,
          h,
          v
        ), e === wt && (at = wt = null, nt = 0), Dh = t, yf = e, mf = a, R0 = f, A0 = o, hb = i, (t.subtreeFlags & 10256) !== 0 || (t.flags & 10256) !== 0 ? (e.callbackNode = null, e.callbackPriority = 0, $y(Fo, function() {
          return pr(), null;
        })) : (e.callbackNode = null, e.callbackPriority = 0), jv = sh(), i = (t.flags & 13878) !== 0, (t.subtreeFlags & 13878) !== 0 || i) {
          i = Y.T, Y.T = null, o = Ue.p, Ue.p = Bl, d = Et, Et |= Iu;
          try {
            od(e, t, a);
          } finally {
            Et = d, Ue.p = o, Y.T = i;
          }
        }
        na = rb, Wn(), Sd(), ov();
      }
    }
    function Wn() {
      if (na === rb) {
        na = Pr;
        var e = yf, t = Dh, a = mf, i = (t.flags & 13878) !== 0;
        if ((t.subtreeFlags & 13878) !== 0 || i) {
          i = Y.T, Y.T = null;
          var o = Ue.p;
          Ue.p = Bl;
          var f = Et;
          Et |= Iu;
          try {
            Th = a, Eh = e, By(t, e), Eh = Th = null, a = w0;
            var d = jp(e.containerInfo), h = a.focusedElem, v = a.selectionRange;
            if (d !== h && h && h.ownerDocument && Yp(
              h.ownerDocument.documentElement,
              h
            )) {
              if (v !== null && Kh(h)) {
                var b = v.start, q = v.end;
                if (q === void 0 && (q = b), "selectionStart" in h)
                  h.selectionStart = b, h.selectionEnd = Math.min(
                    q,
                    h.value.length
                  );
                else {
                  var L = h.ownerDocument || document, x = L && L.defaultView || window;
                  if (x.getSelection) {
                    var V = x.getSelection(), ye = h.textContent.length, Ce = Math.min(
                      v.start,
                      ye
                    ), qt = v.end === void 0 ? Ce : Math.min(v.end, ye);
                    !V.extend && Ce > qt && (d = qt, qt = Ce, Ce = d);
                    var ct = Zh(
                      h,
                      Ce
                    ), T = Zh(
                      h,
                      qt
                    );
                    if (ct && T && (V.rangeCount !== 1 || V.anchorNode !== ct.node || V.anchorOffset !== ct.offset || V.focusNode !== T.node || V.focusOffset !== T.offset)) {
                      var E = L.createRange();
                      E.setStart(ct.node, ct.offset), V.removeAllRanges(), Ce > qt ? (V.addRange(E), V.extend(T.node, T.offset)) : (E.setEnd(T.node, T.offset), V.addRange(E));
                    }
                  }
                }
              }
              for (L = [], V = h; V = V.parentNode; )
                V.nodeType === 1 && L.push({
                  element: V,
                  left: V.scrollLeft,
                  top: V.scrollTop
                });
              for (typeof h.focus == "function" && h.focus(), h = 0; h < L.length; h++) {
                var A = L[h];
                A.element.scrollLeft = A.left, A.element.scrollTop = A.top;
              }
            }
            yg = !!N0, w0 = N0 = null;
          } finally {
            Et = f, Ue.p = o, Y.T = i;
          }
        }
        e.current = t, na = sb;
      }
    }
    function Sd() {
      if (na === sb) {
        na = Pr;
        var e = yf, t = Dh, a = mf, i = (t.flags & 8772) !== 0;
        if ((t.subtreeFlags & 8772) !== 0 || i) {
          i = Y.T, Y.T = null;
          var o = Ue.p;
          Ue.p = Bl;
          var f = Et;
          Et |= Iu;
          try {
            fe !== null && typeof fe.markLayoutEffectsStarted == "function" && fe.markLayoutEffectsStarted(a), Th = a, Eh = e, wy(
              e,
              t.alternate,
              t
            ), Eh = Th = null, fe !== null && typeof fe.markLayoutEffectsStopped == "function" && fe.markLayoutEffectsStopped();
          } finally {
            Et = f, Ue.p = o, Y.T = i;
          }
        }
        na = db;
      }
    }
    function ov() {
      if (na === IS || na === db) {
        na = Pr, Yg();
        var e = yf, t = Dh, a = mf, i = hb, o = (t.subtreeFlags & 10256) !== 0 || (t.flags & 10256) !== 0;
        o ? na = E0 : (na = Pr, Dh = yf = null, Fn(e, e.pendingLanes), es = 0, yp = null);
        var f = e.pendingLanes;
        if (f === 0 && (hf = null), o || wo(e), o = lo(a), t = t.stateNode, ql && typeof ql.onCommitFiberRoot == "function")
          try {
            var d = (t.current.flags & 128) === 128;
            switch (o) {
              case Bl:
                var h = Xd;
                break;
              case Rn:
                h = xr;
                break;
              case Zu:
                h = Fo;
                break;
              case Jd:
                h = Hr;
                break;
              default:
                h = Fo;
            }
            ql.onCommitFiberRoot(
              Ni,
              t,
              h,
              d
            );
          } catch (L) {
            ga || (ga = !0, console.error(
              "React instrumentation encountered an error: %s",
              L
            ));
          }
        if (It && e.memoizedUpdaters.clear(), Gy(), i !== null) {
          d = Y.T, h = Ue.p, Ue.p = Bl, Y.T = null;
          try {
            var v = e.onRecoverableError;
            for (t = 0; t < i.length; t++) {
              var b = i[t], q = fv(b.stack);
              he(
                b.source,
                v,
                b.value,
                q
              );
            }
          } finally {
            Y.T = d, Ue.p = h;
          }
        }
        (mf & 3) !== 0 && Ho(), Wa(e), f = e.pendingLanes, (a & 4194090) !== 0 && (f & 42) !== 0 ? (Lv = !0, e === O0 ? hp++ : (hp = 0, O0 = e)) : hp = 0, zc(0), xe();
      }
    }
    function fv(e) {
      return e = { componentStack: e }, Object.defineProperty(e, "digest", {
        get: function() {
          console.error(
            'You are accessing "digest" from the errorInfo object passed to onRecoverableError. This property is no longer provided as part of errorInfo but can be accessed as a property of the Error instance itself.'
          );
        }
      }), e;
    }
    function Fn(e, t) {
      (e.pooledCacheLanes &= t) === 0 && (t = e.pooledCache, t != null && (e.pooledCache = null, wn(t)));
    }
    function Ho(e) {
      return Wn(), Sd(), ov(), pr();
    }
    function pr() {
      if (na !== E0) return !1;
      var e = yf, t = R0;
      R0 = 0;
      var a = lo(mf), i = Zu > a ? Zu : a;
      a = Y.T;
      var o = Ue.p;
      try {
        Ue.p = i, Y.T = null, i = A0, A0 = null;
        var f = yf, d = mf;
        if (na = Pr, Dh = yf = null, mf = 0, (Et & (Ba | Iu)) !== An)
          throw Error("Cannot flush passive effects while already rendering.");
        D0 = !0, Pv = !1, fe !== null && typeof fe.markPassiveEffectsStarted == "function" && fe.markPassiveEffectsStarted(d);
        var h = Et;
        if (Et |= Iu, Yy(f.current), rr(
          f,
          f.current,
          d,
          i
        ), fe !== null && typeof fe.markPassiveEffectsStopped == "function" && fe.markPassiveEffectsStopped(), wo(f), Et = h, zc(0, !1), Pv ? f === yp ? es++ : (es = 0, yp = f) : es = 0, Pv = D0 = !1, ql && typeof ql.onPostCommitFiberRoot == "function")
          try {
            ql.onPostCommitFiberRoot(Ni, f);
          } catch (b) {
            ga || (ga = !0, console.error(
              "React instrumentation encountered an error: %s",
              b
            ));
          }
        var v = f.current.stateNode;
        return v.effectDuration = 0, v.passiveEffectDuration = 0, !0;
      } finally {
        Ue.p = o, Y.T = a, Fn(e, t);
      }
    }
    function No(e, t, a) {
      t = Oa(a, t), t = Vl(e.stateNode, t, 2), e = yn(e, t, 2), e !== null && (Su(e, 2), Wa(e));
    }
    function Me(e, t, a) {
      if (zh = !1, e.tag === 3)
        No(e, e, a);
      else {
        for (; t !== null; ) {
          if (t.tag === 3) {
            No(
              t,
              e,
              a
            );
            return;
          }
          if (t.tag === 1) {
            var i = t.stateNode;
            if (typeof t.type.getDerivedStateFromError == "function" || typeof i.componentDidCatch == "function" && (hf === null || !hf.has(i))) {
              e = Oa(a, e), a = Kt(2), i = yn(t, a, 2), i !== null && (er(
                a,
                i,
                t,
                e
              ), Su(i, 2), Wa(i));
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
    function Ky(e, t, a) {
      var i = e.pingCache;
      if (i === null) {
        i = e.pingCache = new kS();
        var o = /* @__PURE__ */ new Set();
        i.set(t, o);
      } else
        o = i.get(t), o === void 0 && (o = /* @__PURE__ */ new Set(), i.set(t, o));
      o.has(a) || (b0 = !0, o.add(a), i = Cg.bind(null, e, t, a), It && qo(e, a), t.then(i, i));
    }
    function Cg(e, t, a) {
      var i = e.pingCache;
      i !== null && i.delete(t), e.pingedLanes |= e.suspendedLanes & a, e.warmLanes &= ~a, Ly() && Y.actQueue === null && console.error(
        `A suspended resource finished loading inside a test, but the event was not wrapped in act(...).

When testing, code that resolves suspended data should be wrapped into act(...):

act(() => {
  /* finish loading suspended data */
});
/* assert on the output */

This ensures that you're testing the behavior the user would see in the browser. Learn more at https://react.dev/link/wrap-tests-with-act`
      ), wt === e && (nt & a) === a && (il === $r || il === p0 && (nt & 62914560) === nt && iu() - T0 < ob ? (Et & Ba) === An && Dc(e, 0) : S0 |= a, Ir === nt && (Ir = 0)), Wa(e);
    }
    function Jy(e, t) {
      t === 0 && (t = Un()), e = ca(e, t), e !== null && (Su(e, t), Wa(e));
    }
    function vr(e) {
      var t = e.memoizedState, a = 0;
      t !== null && (a = t.retryLane), Jy(e, a);
    }
    function Oi(e, t) {
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
      i !== null && i.delete(t), Jy(e, a);
    }
    function Td(e, t, a) {
      if ((t.subtreeFlags & 67117056) !== 0)
        for (t = t.child; t !== null; ) {
          var i = e, o = t, f = o.type === Ko;
          f = a || f, o.tag !== 22 ? o.flags & 67108864 ? f && he(
            o,
            ky,
            i,
            o,
            (o.mode & n1) === Gt
          ) : Td(
            i,
            o,
            f
          ) : o.memoizedState === null && (f && o.flags & 8192 ? he(
            o,
            ky,
            i,
            o
          ) : o.subtreeFlags & 67108864 && he(
            o,
            Td,
            i,
            o,
            f
          )), t = t.sibling;
        }
    }
    function ky(e, t) {
      var a = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : !0;
      oe(!0);
      try {
        _a(t), a && hr(t), wu(e, t.alternate, t, !1), a && rd(e, t, 0, null, !1, 0);
      } finally {
        oe(!1);
      }
    }
    function wo(e) {
      var t = !0;
      e.current.mode & (Ta | ku) || (t = !1), Td(
        e,
        e.current,
        t
      );
    }
    function Tn(e) {
      if ((Et & Ba) === An) {
        var t = e.tag;
        if (t === 3 || t === 1 || t === 0 || t === 11 || t === 14 || t === 15) {
          if (t = de(e) || "ReactComponent", eg !== null) {
            if (eg.has(t)) return;
            eg.add(t);
          } else eg = /* @__PURE__ */ new Set([t]);
          he(e, function() {
            console.error(
              "Can't perform a React state update on a component that hasn't mounted yet. This indicates that you have a side-effect in your render function that asynchronously later calls tries to update the component. Move this work to useEffect instead."
            );
          });
        }
      }
    }
    function qo(e, t) {
      It && e.memoizedUpdaters.forEach(function(a) {
        ja(e, a, t);
      });
    }
    function $y(e, t) {
      var a = Y.actQueue;
      return a !== null ? (a.push(t), tT) : Vd(e, t);
    }
    function rv(e) {
      Ly() && Y.actQueue === null && he(e, function() {
        console.error(
          `An update to %s inside a test was not wrapped in act(...).

When testing, code that causes React state updates should be wrapped into act(...):

act(() => {
  /* fire events that update state */
});
/* assert on the output */

This ensures that you're testing the behavior the user would see in the browser. Learn more at https://react.dev/link/wrap-tests-with-act`,
          de(e)
        );
      });
    }
    function Wa(e) {
      e !== Mh && e.next === null && (Mh === null ? tg = Mh = e : Mh = Mh.next = e), lg = !0, Y.actQueue !== null ? M0 || (M0 = !0, hl()) : z0 || (z0 = !0, hl());
    }
    function zc(e, t) {
      if (!_0 && lg) {
        _0 = !0;
        do
          for (var a = !1, i = tg; i !== null; ) {
            if (e !== 0) {
              var o = i.pendingLanes;
              if (o === 0) var f = 0;
              else {
                var d = i.suspendedLanes, h = i.pingedLanes;
                f = (1 << 31 - Zl(42 | e) + 1) - 1, f &= o & ~(d & ~h), f = f & 201326741 ? f & 201326741 | 1 : f ? f | 2 : 0;
              }
              f !== 0 && (a = !0, Ad(i, f));
            } else
              f = nt, f = vl(
                i,
                i === wt ? f : 0,
                i.cancelPendingCommit !== null || i.timeoutHandle !== as
              ), (f & 3) === 0 || Pu(i, f) || (a = !0, Ad(i, f));
            i = i.next;
          }
        while (a);
        _0 = !1;
      }
    }
    function Ed() {
      Rd();
    }
    function Rd() {
      lg = M0 = z0 = !1;
      var e = 0;
      ts !== 0 && (Bo() && (e = ts), ts = 0);
      for (var t = iu(), a = null, i = tg; i !== null; ) {
        var o = i.next, f = In(i, t);
        f === 0 ? (i.next = null, a === null ? tg = o : a.next = o, o === null && (Mh = a)) : (a = i, (e !== 0 || (f & 3) !== 0) && (lg = !0)), i = o;
      }
      zc(e);
    }
    function In(e, t) {
      for (var a = e.suspendedLanes, i = e.pingedLanes, o = e.expirationTimes, f = e.pendingLanes & -62914561; 0 < f; ) {
        var d = 31 - Zl(f), h = 1 << d, v = o[d];
        v === -1 ? ((h & a) === 0 || (h & i) !== 0) && (o[d] = cs(h, t)) : v <= t && (e.expiredLanes |= h), f &= ~h;
      }
      if (t = wt, a = nt, a = vl(
        e,
        e === t ? a : 0,
        e.cancelPendingCommit !== null || e.timeoutHandle !== as
      ), i = e.callbackNode, a === 0 || e === t && (_t === Wr || _t === Fr) || e.cancelPendingCommit !== null)
        return i !== null && Od(i), e.callbackNode = null, e.callbackPriority = 0;
      if ((a & 3) === 0 || Pu(e, a)) {
        if (t = a & -a, t !== e.callbackPriority || Y.actQueue !== null && i !== U0)
          Od(i);
        else return t;
        switch (lo(a)) {
          case Bl:
          case Rn:
            a = xr;
            break;
          case Zu:
            a = Fo;
            break;
          case Jd:
            a = Hr;
            break;
          default:
            a = Fo;
        }
        return i = kt.bind(null, e), Y.actQueue !== null ? (Y.actQueue.push(i), a = U0) : a = Vd(a, i), e.callbackPriority = t, e.callbackNode = a, t;
      }
      return i !== null && Od(i), e.callbackPriority = 2, e.callbackNode = null, 2;
    }
    function kt(e, t) {
      if (Lv = Gv = !1, na !== Pr && na !== E0)
        return e.callbackNode = null, e.callbackPriority = 0, null;
      var a = e.callbackNode;
      if (Ho() && e.callbackNode !== a)
        return null;
      var i = nt;
      return i = vl(
        e,
        e === wt ? i : 0,
        e.cancelPendingCommit !== null || e.timeoutHandle !== as
      ), i === 0 ? null : (dl(
        e,
        i,
        t
      ), In(e, iu()), e.callbackNode != null && e.callbackNode === a ? kt.bind(null, e) : null);
    }
    function Ad(e, t) {
      if (Ho()) return null;
      Gv = Lv, Lv = !1, dl(e, t, !0);
    }
    function Od(e) {
      e !== U0 && e !== null && Bg(e);
    }
    function hl() {
      Y.actQueue !== null && Y.actQueue.push(function() {
        return Rd(), null;
      }), sT(function() {
        (Et & (Ba | Iu)) !== An ? Vd(
          Xd,
          Ed
        ) : Rd();
      });
    }
    function Wy() {
      return ts === 0 && (ts = Je()), ts;
    }
    function Fy(e) {
      return e == null || typeof e == "symbol" || typeof e == "boolean" ? null : typeof e == "function" ? e : (J(e, "action"), fo("" + e));
    }
    function Iy(e, t) {
      var a = t.ownerDocument.createElement("input");
      return a.name = t.name, a.value = t.value, e.id && a.setAttribute("form", e.id), t.parentNode.insertBefore(a, t), e = new FormData(e), a.parentNode.removeChild(a), e;
    }
    function Yt(e, t, a, i, o) {
      if (t === "submit" && a && a.stateNode === o) {
        var f = Fy(
          (o[ba] || null).action
        ), d = i.submitter;
        d && (t = (t = d[ba] || null) ? Fy(t.formAction) : d.getAttribute("formAction"), t !== null && (f = t, d = null));
        var h = new Ee(
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
                  if (ts !== 0) {
                    var v = d ? Iy(
                      o,
                      d
                    ) : new FormData(o), b = {
                      pending: !0,
                      data: v,
                      method: o.method,
                      action: f
                    };
                    Object.freeze(b), yc(
                      a,
                      b,
                      null,
                      v
                    );
                  }
                } else
                  typeof f == "function" && (h.preventDefault(), v = d ? Iy(
                    o,
                    d
                  ) : new FormData(o), b = {
                    pending: !0,
                    data: v,
                    method: o.method,
                    action: f
                  }, Object.freeze(b), yc(
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
    function xl(e, t, a) {
      e.currentTarget = a;
      try {
        t(e);
      } catch (i) {
        s0(i);
      }
      e.currentTarget = null;
    }
    function Pn(e, t) {
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
              v !== null ? he(
                v,
                xl,
                f,
                h,
                b
              ) : xl(f, h, b), o = v;
            }
          else
            for (d = 0; d < i.length; d++) {
              if (h = i[d], v = h.instance, b = h.currentTarget, h = h.listener, v !== o && f.isPropagationStopped())
                break e;
              v !== null ? he(
                v,
                xl,
                f,
                h,
                b
              ) : xl(f, h, b), o = v;
            }
        }
      }
    }
    function et(e, t) {
      C0.has(e) || console.error(
        'Did not expect a listenToNonDelegatedEvent() call for "%s". This is a bug in React. Please file an issue.',
        e
      );
      var a = t[Om];
      a === void 0 && (a = t[Om] = /* @__PURE__ */ new Set());
      var i = e + "__bubble";
      a.has(i) || (zd(t, e, 2, !1), a.add(i));
    }
    function Dd(e, t, a) {
      C0.has(e) && !t && console.error(
        'Did not expect a listenToNativeEvent() call for "%s" in the bubble phase. This is a bug in React. Please file an issue.',
        e
      );
      var i = 0;
      t && (i |= 4), zd(
        a,
        e,
        i,
        t
      );
    }
    function Py(e) {
      if (!e[ag]) {
        e[ag] = !0, Dv.forEach(function(a) {
          a !== "selectionchange" && (C0.has(a) || Dd(a, !1, e), Dd(a, !0, e));
        });
        var t = e.nodeType === 9 ? e : e.ownerDocument;
        t === null || t[ag] || (t[ag] = !0, Dd("selectionchange", !1, t));
      }
    }
    function zd(e, t, a, i) {
      switch (Yd(t)) {
        case Bl:
          var o = Ng;
          break;
        case Rn:
          o = Bd;
          break;
        default:
          o = _i;
      }
      a = o.bind(
        null,
        t,
        a,
        e
      ), o = void 0, !C || t !== "touchstart" && t !== "touchmove" && t !== "wheel" || (o = !0), i ? o !== void 0 ? e.addEventListener(t, a, {
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
    function Il(e, t, a, i, o) {
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
              if (d = ia(h), d === null) return;
              if (v = d.tag, v === 5 || v === 6 || v === 26 || v === 27) {
                i = f = d;
                continue e;
              }
              h = h.parentNode;
            }
          }
          i = i.return;
        }
      gs(function() {
        var b = f, q = ec(a), L = [];
        e: {
          var x = a1.get(e);
          if (x !== void 0) {
            var V = Ee, ye = e;
            switch (e) {
              case "keypress":
                if (ro(a) === 0) break e;
              case "keydown":
              case "keyup":
                V = gS;
                break;
              case "focusin":
                ye = "focus", V = ft;
                break;
              case "focusout":
                ye = "blur", V = ft;
                break;
              case "beforeblur":
              case "afterblur":
                V = ft;
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
                V = We;
                break;
              case "drag":
              case "dragend":
              case "dragenter":
              case "dragexit":
              case "dragleave":
              case "dragover":
              case "dragstart":
              case "drop":
                V = _e;
                break;
              case "touchcancel":
              case "touchend":
              case "touchmove":
              case "touchstart":
                V = TS;
                break;
              case P0:
              case e1:
              case t1:
                V = Vg;
                break;
              case l1:
                V = RS;
                break;
              case "scroll":
              case "scrollend":
                V = M;
                break;
              case "wheel":
                V = OS;
                break;
              case "copy":
              case "cut":
              case "paste":
                V = sS;
                break;
              case "gotpointercapture":
              case "lostpointercapture":
              case "pointercancel":
              case "pointerdown":
              case "pointermove":
              case "pointerout":
              case "pointerover":
              case "pointerup":
                V = Z0;
                break;
              case "toggle":
              case "beforetoggle":
                V = zS;
            }
            var Ce = (t & 4) !== 0, qt = !Ce && (e === "scroll" || e === "scrollend"), ct = Ce ? x !== null ? x + "Capture" : null : x;
            Ce = [];
            for (var T = b, E; T !== null; ) {
              var A = T;
              if (E = A.stateNode, A = A.tag, A !== 5 && A !== 26 && A !== 27 || E === null || ct === null || (A = Ru(T, ct), A != null && Ce.push(
                Pl(
                  T,
                  A,
                  E
                )
              )), qt) break;
              T = T.return;
            }
            0 < Ce.length && (x = new V(
              x,
              ye,
              null,
              a,
              q
            ), L.push({
              event: x,
              listeners: Ce
            }));
          }
        }
        if ((t & 7) === 0) {
          e: {
            if (x = e === "mouseover" || e === "pointerover", V = e === "mouseout" || e === "pointerout", x && a !== r && (ye = a.relatedTarget || a.fromElement) && (ia(ye) || ye[qi]))
              break e;
            if ((V || x) && (x = q.window === q ? q : (x = q.ownerDocument) ? x.defaultView || x.parentWindow : window, V ? (ye = a.relatedTarget || a.toElement, V = b, ye = ye ? ia(ye) : null, ye !== null && (qt = Pe(ye), Ce = ye.tag, ye !== qt || Ce !== 5 && Ce !== 27 && Ce !== 6) && (ye = null)) : (V = null, ye = b), V !== ye)) {
              if (Ce = We, A = "onMouseLeave", ct = "onMouseEnter", T = "mouse", (e === "pointerout" || e === "pointerover") && (Ce = Z0, A = "onPointerLeave", ct = "onPointerEnter", T = "pointer"), qt = V == null ? x : cn(V), E = ye == null ? x : cn(ye), x = new Ce(
                A,
                T + "leave",
                V,
                a,
                q
              ), x.target = qt, x.relatedTarget = E, A = null, ia(q) === b && (Ce = new Ce(
                ct,
                T + "enter",
                ye,
                a,
                q
              ), Ce.target = E, Ce.relatedTarget = qt, A = Ce), qt = A, V && ye)
                t: {
                  for (Ce = V, ct = ye, T = 0, E = Ce; E; E = El(E))
                    T++;
                  for (E = 0, A = ct; A; A = El(A))
                    E++;
                  for (; 0 < T - E; )
                    Ce = El(Ce), T--;
                  for (; 0 < E - T; )
                    ct = El(ct), E--;
                  for (; T--; ) {
                    if (Ce === ct || ct !== null && Ce === ct.alternate)
                      break t;
                    Ce = El(Ce), ct = El(ct);
                  }
                  Ce = null;
                }
              else Ce = null;
              V !== null && em(
                L,
                x,
                V,
                Ce,
                !1
              ), ye !== null && qt !== null && em(
                L,
                qt,
                ye,
                Ce,
                !0
              );
            }
          }
          e: {
            if (x = b ? cn(b) : window, V = x.nodeName && x.nodeName.toLowerCase(), V === "select" || V === "input" && x.type === "file")
              var Q = Xh;
            else if (Np(x))
              if (F0)
                Q = zg;
              else {
                Q = Qh;
                var ae = Og;
              }
            else
              V = x.nodeName, !V || V.toLowerCase() !== "input" || x.type !== "checkbox" && x.type !== "radio" ? b && Pi(b.elementType) && (Q = Xh) : Q = Dg;
            if (Q && (Q = Q(e, b))) {
              Es(
                L,
                Q,
                a,
                q
              );
              break e;
            }
            ae && ae(e, x, b), e === "focusout" && b && x.type === "number" && b.memoizedProps.value != null && ds(x, "number", x.value);
          }
          switch (ae = b ? cn(b) : window, e) {
            case "focusin":
              (Np(ae) || ae.contentEditable === "true") && (ah = ae, Qg = b, Zm = null);
              break;
            case "focusout":
              Zm = Qg = ah = null;
              break;
            case "mousedown":
              Zg = !0;
              break;
            case "contextmenu":
            case "mouseup":
            case "dragend":
              Zg = !1, Gp(
                L,
                a,
                q
              );
              break;
            case "selectionchange":
              if (CS) break;
            case "keydown":
            case "keyup":
              Gp(
                L,
                a,
                q
              );
          }
          var Xe;
          if (Xg)
            e: {
              switch (e) {
                case "compositionstart":
                  var me = "onCompositionStart";
                  break e;
                case "compositionend":
                  me = "onCompositionEnd";
                  break e;
                case "compositionupdate":
                  me = "onCompositionUpdate";
                  break e;
              }
              me = void 0;
            }
          else
            lh ? Fl(e, a) && (me = "onCompositionEnd") : e === "keydown" && a.keyCode === K0 && (me = "onCompositionStart");
          me && (J0 && a.locale !== "ko" && (lh || me !== "onCompositionStart" ? me === "onCompositionEnd" && lh && (Xe = Au()) : (W = q, N = "value" in W ? W.value : W.textContent, lh = !0)), ae = gr(
            b,
            me
          ), 0 < ae.length && (me = new Q0(
            me,
            e,
            null,
            a,
            q
          ), L.push({
            event: me,
            listeners: ae
          }), Xe ? me.data = Xe : (Xe = ui(a), Xe !== null && (me.data = Xe)))), (Xe = _S ? Ts(e, a) : Cf(e, a)) && (me = gr(
            b,
            "onBeforeInput"
          ), 0 < me.length && (ae = new hS(
            "onBeforeInput",
            "beforeinput",
            null,
            a,
            q
          ), L.push({
            event: ae,
            listeners: me
          }), ae.data = Xe)), Yt(
            L,
            e,
            b,
            a,
            q
          );
        }
        Pn(L, t);
      });
    }
    function Pl(e, t, a) {
      return {
        instance: e,
        listener: t,
        currentTarget: a
      };
    }
    function gr(e, t) {
      for (var a = t + "Capture", i = []; e !== null; ) {
        var o = e, f = o.stateNode;
        if (o = o.tag, o !== 5 && o !== 26 && o !== 27 || f === null || (o = Ru(e, a), o != null && i.unshift(
          Pl(e, o, f)
        ), o = Ru(e, t), o != null && i.push(
          Pl(e, o, f)
        )), e.tag === 3) return i;
        e = e.return;
      }
      return [];
    }
    function El(e) {
      if (e === null) return null;
      do
        e = e.return;
      while (e && e.tag !== 5 && e.tag !== 27);
      return e || null;
    }
    function em(e, t, a, i, o) {
      for (var f = t._reactName, d = []; a !== null && a !== i; ) {
        var h = a, v = h.alternate, b = h.stateNode;
        if (h = h.tag, v !== null && v === i) break;
        h !== 5 && h !== 26 && h !== 27 || b === null || (v = b, o ? (b = Ru(a, f), b != null && d.unshift(
          Pl(a, b, v)
        )) : o || (b = Ru(a, f), b != null && d.push(
          Pl(a, b, v)
        ))), a = a.return;
      }
      d.length !== 0 && e.push({ event: t, listeners: d });
    }
    function eu(e, t) {
      oo(e, t), e !== "input" && e !== "textarea" && e !== "select" || t == null || t.value !== null || Lm || (Lm = !0, e === "select" && t.multiple ? console.error(
        "`value` prop on `%s` should not be null. Consider using an empty array when `multiple` is set to `true` to clear the component or `undefined` for uncontrolled components.",
        e
      ) : console.error(
        "`value` prop on `%s` should not be null. Consider using an empty string to clear the component or `undefined` for uncontrolled components.",
        e
      ));
      var a = {
        registrationNameDependencies: tn,
        possibleRegistrationNames: qc
      };
      Pi(e) || typeof t.is == "string" || Gh(e, t, a), t.contentEditable && !t.suppressContentEditableWarning && t.children != null && console.error(
        "A component is `contentEditable` and contains `children` managed by React. It is now your responsibility to guarantee that none of those nodes are unexpectedly modified or duplicated. This is probably not intentional."
      );
    }
    function jt(e, t, a, i) {
      t !== a && (a = Hl(a), Hl(t) !== a && (i[e] = t));
    }
    function Di(e, t, a) {
      t.forEach(function(i) {
        a[lm(i)] = i === "style" ? _c(e) : e.getAttribute(i);
      });
    }
    function Fa(e, t) {
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
    function Md(e, t) {
      return e = e.namespaceURI === Gr || e.namespaceURI === nf ? e.ownerDocument.createElementNS(
        e.namespaceURI,
        e.tagName
      ) : e.ownerDocument.createElement(e.tagName), e.innerHTML = t, e.innerHTML;
    }
    function Hl(e) {
      return g(e) && (console.error(
        "The provided HTML markup uses a value of unsupported type %s. This value must be coerced to a string before using it here.",
        ge(e)
      ), w(e)), (typeof e == "string" ? e : "" + e).replace(lT, `
`).replace(aT, "");
    }
    function tm(e, t) {
      return t = Hl(t), Hl(e) === t;
    }
    function Bu() {
    }
    function ht(e, t, a, i, o, f) {
      switch (a) {
        case "children":
          typeof i == "string" ? (_f(i, t, !1), t === "body" || t === "textarea" && i === "" || Ii(e, i)) : (typeof i == "number" || typeof i == "bigint") && (_f("" + i, t, !1), t !== "body" && Ii(e, "" + i));
          break;
        case "className":
          Ye(e, "class", i);
          break;
        case "tabIndex":
          Ye(e, "tabindex", i);
          break;
        case "dir":
        case "role":
        case "viewBox":
        case "width":
        case "height":
          Ye(e, a, i);
          break;
        case "style":
          Uf(e, i, f);
          break;
        case "data":
          if (t !== "object") {
            Ye(e, "data", i);
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
          J(i, a), i = fo("" + i), e.setAttribute(a, i);
          break;
        case "action":
        case "formAction":
          if (i != null && (t === "form" ? a === "formAction" ? console.error(
            "You can only pass the formAction prop to <input> or <button>. Use the action prop on <form>."
          ) : typeof i == "function" && (o.encType == null && o.method == null || ig || (ig = !0, console.error(
            "Cannot specify a encType or method for a form that specifies a function as the action. React provides those automatically. They will get overridden."
          )), o.target == null || ug || (ug = !0, console.error(
            "Cannot specify a target for a form that specifies a function as the action. The function will always be executed in the same window."
          ))) : t === "input" || t === "button" ? a === "action" ? console.error(
            "You can only pass the action prop to <form>. Use the formAction prop on <input> or <button>."
          ) : t !== "input" || o.type === "submit" || o.type === "image" || ng ? t !== "button" || o.type == null || o.type === "submit" || ng ? typeof i == "function" && (o.name == null || gb || (gb = !0, console.error(
            'Cannot specify a "name" prop for a button that specifies a function as a formAction. React needs it to encode which action should be invoked. It will get overridden.'
          )), o.formEncType == null && o.formMethod == null || ig || (ig = !0, console.error(
            "Cannot specify a formEncType or formMethod for a button that specifies a function as a formAction. React provides those automatically. They will get overridden."
          )), o.formTarget == null || ug || (ug = !0, console.error(
            "Cannot specify a formTarget for a button that specifies a function as a formAction. The function will always be executed in the same window."
          ))) : (ng = !0, console.error(
            'A button can only specify a formAction along with type="submit" or no type.'
          )) : (ng = !0, console.error(
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
          J(i, a), i = fo("" + i), e.setAttribute(a, i);
          break;
        case "onClick":
          i != null && (typeof i != "function" && Fa(a, i), e.onclick = Bu);
          break;
        case "onScroll":
          i != null && (typeof i != "function" && Fa(a, i), et("scroll", e));
          break;
        case "onScrollEnd":
          i != null && (typeof i != "function" && Fa(a, i), et("scrollend", e));
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
          J(i, a), a = fo("" + i), e.setAttributeNS(ls, "xlink:href", a);
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
          i !== "" || cg[a] || (cg[a] = !0, console.error(
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
          et("beforetoggle", e), et("toggle", e), ut(e, "popover", i);
          break;
        case "xlinkActuate":
          al(
            e,
            ls,
            "xlink:actuate",
            i
          );
          break;
        case "xlinkArcrole":
          al(
            e,
            ls,
            "xlink:arcrole",
            i
          );
          break;
        case "xlinkRole":
          al(
            e,
            ls,
            "xlink:role",
            i
          );
          break;
        case "xlinkShow":
          al(
            e,
            ls,
            "xlink:show",
            i
          );
          break;
        case "xlinkTitle":
          al(
            e,
            ls,
            "xlink:title",
            i
          );
          break;
        case "xlinkType":
          al(
            e,
            ls,
            "xlink:type",
            i
          );
          break;
        case "xmlBase":
          al(
            e,
            x0,
            "xml:base",
            i
          );
          break;
        case "xmlLang":
          al(
            e,
            x0,
            "xml:lang",
            i
          );
          break;
        case "xmlSpace":
          al(
            e,
            x0,
            "xml:space",
            i
          );
          break;
        case "is":
          f != null && console.error(
            'Cannot update the "is" prop after it has been initialized.'
          ), ut(e, "is", i);
          break;
        case "innerText":
        case "textContent":
          break;
        case "popoverTarget":
          bb || i == null || typeof i != "object" || (bb = !0, console.error(
            "The `popoverTarget` prop expects the ID of an Element as a string. Received %s instead.",
            i
          ));
        default:
          !(2 < a.length) || a[0] !== "o" && a[0] !== "O" || a[1] !== "n" && a[1] !== "N" ? (a = vs(a), ut(e, a, i)) : tn.hasOwnProperty(a) && i != null && typeof i != "function" && Fa(a, i);
      }
    }
    function Mc(e, t, a, i, o, f) {
      switch (a) {
        case "style":
          Uf(e, i, f);
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
          typeof i == "string" ? Ii(e, i) : (typeof i == "number" || typeof i == "bigint") && Ii(e, "" + i);
          break;
        case "onScroll":
          i != null && (typeof i != "function" && Fa(a, i), et("scroll", e));
          break;
        case "onScrollEnd":
          i != null && (typeof i != "function" && Fa(a, i), et("scrollend", e));
          break;
        case "onClick":
          i != null && (typeof i != "function" && Fa(a, i), e.onclick = Bu);
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
          if (tn.hasOwnProperty(a))
            i != null && typeof i != "function" && Fa(a, i);
          else
            e: {
              if (a[0] === "o" && a[1] === "n" && (o = a.endsWith("Capture"), t = a.slice(2, o ? a.length - 7 : void 0), f = e[ba] || null, f = f != null ? f[a] : null, typeof f == "function" && e.removeEventListener(t, f, o), typeof i == "function")) {
                typeof f != "function" && f !== null && (a in e ? e[a] = null : e.hasAttribute(a) && e.removeAttribute(a)), e.addEventListener(t, i, o);
                break e;
              }
              a in e ? e[a] = i : i === !0 ? e.setAttribute(a, "") : ut(e, a, i);
            }
      }
    }
    function $t(e, t, a) {
      switch (eu(t, a), t) {
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
          et("error", e), et("load", e);
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
          pe("input", a), et("invalid", e);
          var h = f = d = o = null, v = null, b = null;
          for (i in a)
            if (a.hasOwnProperty(i)) {
              var q = a[i];
              if (q != null)
                switch (i) {
                  case "name":
                    o = q;
                    break;
                  case "type":
                    d = q;
                    break;
                  case "checked":
                    v = q;
                    break;
                  case "defaultChecked":
                    b = q;
                    break;
                  case "value":
                    f = q;
                    break;
                  case "defaultValue":
                    h = q;
                    break;
                  case "children":
                  case "dangerouslySetInnerHTML":
                    if (q != null)
                      throw Error(
                        t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                      );
                    break;
                  default:
                    ht(e, t, i, q, a, null);
                }
            }
          ti(e, a), _p(
            e,
            f,
            h,
            v,
            b,
            d,
            o,
            !1
          ), Tu(e);
          return;
        case "select":
          pe("select", a), et("invalid", e), i = d = f = null;
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
          Df(e, a), t = f, a = d, e.multiple = !!i, t != null ? Eu(e, !!i, t, !1) : a != null && Eu(e, !!i, a, !0);
          return;
        case "textarea":
          pe("textarea", a), et("invalid", e), f = o = i = null;
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
          Cn(e, a), wh(e, i, o, f), Tu(e);
          return;
        case "option":
          Nh(e, a);
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
          et("beforetoggle", e), et("toggle", e), et("cancel", e), et("close", e);
          break;
        case "iframe":
        case "object":
          et("load", e);
          break;
        case "video":
        case "audio":
          for (i = 0; i < mp.length; i++)
            et(mp[i], e);
          break;
        case "image":
          et("error", e), et("load", e);
          break;
        case "details":
          et("toggle", e);
          break;
        case "embed":
        case "source":
        case "link":
          et("error", e), et("load", e);
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
          if (Pi(t)) {
            for (q in a)
              a.hasOwnProperty(q) && (i = a[q], i !== void 0 && Mc(
                e,
                t,
                q,
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
      switch (eu(t, i), t) {
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
          var o = null, f = null, d = null, h = null, v = null, b = null, q = null;
          for (V in a) {
            var L = a[V];
            if (a.hasOwnProperty(V) && L != null)
              switch (V) {
                case "checked":
                  break;
                case "value":
                  break;
                case "defaultValue":
                  v = L;
                default:
                  i.hasOwnProperty(V) || ht(
                    e,
                    t,
                    V,
                    null,
                    i,
                    L
                  );
              }
          }
          for (var x in i) {
            var V = i[x];
            if (L = a[x], i.hasOwnProperty(x) && (V != null || L != null))
              switch (x) {
                case "type":
                  f = V;
                  break;
                case "name":
                  o = V;
                  break;
                case "checked":
                  b = V;
                  break;
                case "defaultChecked":
                  q = V;
                  break;
                case "value":
                  d = V;
                  break;
                case "defaultValue":
                  h = V;
                  break;
                case "children":
                case "dangerouslySetInnerHTML":
                  if (V != null)
                    throw Error(
                      t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                    );
                  break;
                default:
                  V !== L && ht(
                    e,
                    t,
                    x,
                    V,
                    i,
                    L
                  );
              }
          }
          t = a.type === "checkbox" || a.type === "radio" ? a.checked != null : a.value != null, i = i.type === "checkbox" || i.type === "radio" ? i.checked != null : i.value != null, t || !i || vb || (console.error(
            "A component is changing an uncontrolled input to be controlled. This is likely caused by the value changing from undefined to a defined value, which should not happen. Decide between using a controlled or uncontrolled input element for the lifetime of the component. More info: https://react.dev/link/controlled-components"
          ), vb = !0), !t || i || pb || (console.error(
            "A component is changing a controlled input to be uncontrolled. This is likely caused by the value changing from a defined to undefined, which should not happen. Decide between using a controlled or uncontrolled input element for the lifetime of the component. More info: https://react.dev/link/controlled-components"
          ), pb = !0), li(
            e,
            d,
            h,
            v,
            b,
            q,
            f,
            o
          );
          return;
        case "select":
          V = d = h = x = null;
          for (f in a)
            if (v = a[f], a.hasOwnProperty(f) && v != null)
              switch (f) {
                case "value":
                  break;
                case "multiple":
                  V = v;
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
                  x = f;
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
          i = h, t = d, a = V, x != null ? Eu(e, !!t, x, !1) : !!a != !!t && (i != null ? Eu(e, !!t, i, !0) : Eu(e, !!t, t ? [] : "", !1));
          return;
        case "textarea":
          V = x = null;
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
                  x = o;
                  break;
                case "defaultValue":
                  V = o;
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
          hs(e, x, V);
          return;
        case "option":
          for (var ye in a)
            if (x = a[ye], a.hasOwnProperty(ye) && x != null && !i.hasOwnProperty(ye))
              switch (ye) {
                case "selected":
                  e.selected = !1;
                  break;
                default:
                  ht(
                    e,
                    t,
                    ye,
                    null,
                    i,
                    x
                  );
              }
          for (v in i)
            if (x = i[v], V = a[v], i.hasOwnProperty(v) && x !== V && (x != null || V != null))
              switch (v) {
                case "selected":
                  e.selected = x && typeof x != "function" && typeof x != "symbol";
                  break;
                default:
                  ht(
                    e,
                    t,
                    v,
                    x,
                    i,
                    V
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
          for (var Ce in a)
            x = a[Ce], a.hasOwnProperty(Ce) && x != null && !i.hasOwnProperty(Ce) && ht(
              e,
              t,
              Ce,
              null,
              i,
              x
            );
          for (b in i)
            if (x = i[b], V = a[b], i.hasOwnProperty(b) && x !== V && (x != null || V != null))
              switch (b) {
                case "children":
                case "dangerouslySetInnerHTML":
                  if (x != null)
                    throw Error(
                      t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                    );
                  break;
                default:
                  ht(
                    e,
                    t,
                    b,
                    x,
                    i,
                    V
                  );
              }
          return;
        default:
          if (Pi(t)) {
            for (var qt in a)
              x = a[qt], a.hasOwnProperty(qt) && x !== void 0 && !i.hasOwnProperty(qt) && Mc(
                e,
                t,
                qt,
                void 0,
                i,
                x
              );
            for (q in i)
              x = i[q], V = a[q], !i.hasOwnProperty(q) || x === V || x === void 0 && V === void 0 || Mc(
                e,
                t,
                q,
                x,
                i,
                V
              );
            return;
          }
      }
      for (var ct in a)
        x = a[ct], a.hasOwnProperty(ct) && x != null && !i.hasOwnProperty(ct) && ht(e, t, ct, null, i, x);
      for (L in i)
        x = i[L], V = a[L], !i.hasOwnProperty(L) || x === V || x == null && V == null || ht(e, t, L, x, i, V);
    }
    function lm(e) {
      switch (e) {
        case "class":
          return "className";
        case "for":
          return "htmlFor";
        default:
          return e;
      }
    }
    function _c(e) {
      var t = {};
      e = e.style;
      for (var a = 0; a < e.length; a++) {
        var i = e[a];
        t[i] = e.getPropertyValue(i);
      }
      return t;
    }
    function am(e, t, a) {
      if (t != null && typeof t != "object")
        console.error(
          "The `style` prop expects a mapping from style properties to values, not a string. For example, style={{marginRight: spacing + 'em'}} when using JSX."
        );
      else {
        var i, o = i = "", f;
        for (f in t)
          if (t.hasOwnProperty(f)) {
            var d = t[f];
            d != null && typeof d != "boolean" && d !== "" && (f.indexOf("--") === 0 ? (P(d, f), i += o + f + ":" + ("" + d).trim()) : typeof d != "number" || d === 0 || jr.has(f) ? (P(d, f), i += o + f.replace(Ku, "-$1").toLowerCase().replace(Ju, "-ms-") + ":" + ("" + d).trim()) : i += o + f.replace(Ku, "-$1").toLowerCase().replace(Ju, "-ms-") + ":" + d + "px", o = ";");
          }
        i = i || null, t = e.getAttribute("style"), t !== i && (i = Hl(i), Hl(t) !== i && (a.style = _c(e)));
      }
    }
    function ea(e, t, a, i, o, f) {
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
    function nm(e, t, a, i, o, f) {
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
    function um(e, t, a, i, o, f) {
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
    function dv(e, t, a, i, o, f) {
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
            if (J(i, t), a = fo("" + i), e === a)
              return;
        }
      jt(t, e, i, f);
    }
    function Ut(e, t, a, i) {
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
      if (Pi(t)) {
        for (var v in a)
          if (a.hasOwnProperty(v)) {
            var b = a[v];
            if (b != null) {
              if (tn.hasOwnProperty(v))
                typeof b != "function" && Fa(v, b);
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
                    d = e.innerHTML, b = b ? b.__html : void 0, b != null && (b = Md(e, b), jt(
                      v,
                      d,
                      b,
                      o
                    ));
                    continue;
                  case "style":
                    f.delete(v), am(e, b, o);
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
                    f.delete("class"), d = Ge(
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
                    i.context === $c && t !== "svg" && t !== "math" ? f.delete(v.toLowerCase()) : f.delete(v), d = Ge(
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
            if (tn.hasOwnProperty(b))
              typeof v != "function" && Fa(b, v);
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
                  d = e.innerHTML, v = v ? v.__html : void 0, v != null && (v = Md(e, v), d !== v && (o[b] = { __html: d }));
                  continue;
                case "className":
                  ea(
                    e,
                    b,
                    "class",
                    v,
                    f,
                    o
                  );
                  continue;
                case "tabIndex":
                  ea(
                    e,
                    b,
                    "tabindex",
                    v,
                    f,
                    o
                  );
                  continue;
                case "style":
                  f.delete(b), am(e, v, o);
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
                  } else if (d === nT) {
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
                  um(
                    e,
                    b,
                    "contenteditable",
                    v,
                    f,
                    o
                  );
                  continue;
                case "spellCheck":
                  um(
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
                  um(
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
                  nm(
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
                    var q = d = b, L = o;
                    if (f.delete(q), h = h.getAttribute(q), h === null)
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
                      L
                    );
                  }
                  continue;
                case "cols":
                case "rows":
                case "size":
                case "span":
                  e: {
                    if (h = e, q = d = b, L = o, f.delete(q), h = h.getAttribute(q), h === null)
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
                      L
                    );
                  }
                  continue;
                case "rowSpan":
                  dv(
                    e,
                    b,
                    "rowspan",
                    v,
                    f,
                    o
                  );
                  continue;
                case "start":
                  dv(
                    e,
                    b,
                    b,
                    v,
                    f,
                    o
                  );
                  continue;
                case "xHeight":
                  ea(
                    e,
                    b,
                    "x-height",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkActuate":
                  ea(
                    e,
                    b,
                    "xlink:actuate",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkArcrole":
                  ea(
                    e,
                    b,
                    "xlink:arcrole",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkRole":
                  ea(
                    e,
                    b,
                    "xlink:role",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkShow":
                  ea(
                    e,
                    b,
                    "xlink:show",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkTitle":
                  ea(
                    e,
                    b,
                    "xlink:title",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xlinkType":
                  ea(
                    e,
                    b,
                    "xlink:type",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xmlBase":
                  ea(
                    e,
                    b,
                    "xml:base",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xmlLang":
                  ea(
                    e,
                    b,
                    "xml:lang",
                    v,
                    f,
                    o
                  );
                  continue;
                case "xmlSpace":
                  ea(
                    e,
                    b,
                    "xml:space",
                    v,
                    f,
                    o
                  );
                  continue;
                case "inert":
                  v !== "" || cg[b] || (cg[b] = !0, console.error(
                    "Received an empty string for a boolean attribute `%s`. This will treat the attribute as if it were false. Either pass `false` to silence this warning, or pass `true` if you used an empty string in earlier versions of React to indicate this attribute is true.",
                    b
                  )), nm(
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
                    h = vs(b), d = !1, i.context === $c && t !== "svg" && t !== "math" ? f.delete(h.toLowerCase()) : (q = b.toLowerCase(), q = jc.hasOwnProperty(
                      q
                    ) && jc[q] || null, q !== null && q !== b && (d = !0, f.delete(q)), f.delete(h));
                    e: if (q = e, L = h, h = v, we(L))
                      if (q.hasAttribute(L))
                        q = q.getAttribute(
                          L
                        ), J(
                          h,
                          L
                        ), h = q === "" + h ? h : q;
                      else {
                        switch (typeof h) {
                          case "function":
                          case "symbol":
                            break e;
                          case "boolean":
                            if (q = L.toLowerCase().slice(0, 5), q !== "data-" && q !== "aria-")
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
      return 0 < f.size && a.suppressHydrationWarning !== !0 && Di(e, f, o), Object.keys(o).length === 0 ? null : o;
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
    function bt(e) {
      switch (e) {
        case nf:
          return _h;
        case Gr:
          return rg;
        default:
          return $c;
      }
    }
    function ma(e, t) {
      if (e === $c)
        switch (t) {
          case "svg":
            return _h;
          case "math":
            return rg;
          default:
            return $c;
        }
      return e === _h && t === "foreignObject" ? $c : e;
    }
    function tu(e, t) {
      return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.children == "bigint" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null;
    }
    function Bo() {
      var e = window.event;
      return e && e.type === "popstate" ? e === q0 ? !1 : (q0 = e, !0) : (q0 = null, !1);
    }
    function im(e) {
      setTimeout(function() {
        throw e;
      });
    }
    function Yu(e, t, a) {
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
    function Wt(e, t, a, i) {
      sv(e, t, a, i), e[ba] = i;
    }
    function ju(e) {
      Ii(e, "");
    }
    function Uc(e, t, a) {
      e.nodeValue = a;
    }
    function lu(e) {
      return e === "head";
    }
    function Ia(e, t) {
      e.removeChild(t);
    }
    function Yo(e, t) {
      (e.nodeType === 9 ? e.body : e.nodeName === "HTML" ? e.ownerDocument.body : e).removeChild(t);
    }
    function jo(e, t) {
      var a = t, i = 0, o = 0;
      do {
        var f = a.nextSibling;
        if (e.removeChild(a), f && f.nodeType === 8)
          if (a = f.data, a === fg) {
            if (0 < i && 8 > i) {
              a = i;
              var d = e.ownerDocument;
              if (a & iT && Xo(d.documentElement), a & cT && Xo(d.body), a & oT)
                for (a = d.head, Xo(a), d = a.firstChild; d; ) {
                  var h = d.nextSibling, v = d.nodeName;
                  d[ef] || v === "SCRIPT" || v === "STYLE" || v === "LINK" && d.rel.toLowerCase() === "stylesheet" || a.removeChild(d), d = h;
                }
            }
            if (o === 0) {
              e.removeChild(f), Nc(t);
              return;
            }
            o--;
          } else
            a === og || a === kc || a === pp ? o++ : i = a.charCodeAt(0) - 48;
        else i = 0;
        a = f;
      } while (a);
      Nc(t);
    }
    function pa(e) {
      e = e.style, typeof e.setProperty == "function" ? e.setProperty("display", "none", "important") : e.display = "none";
    }
    function cm(e) {
      e.nodeValue = "";
    }
    function om(e, t) {
      t = t[fT], t = t != null && t.hasOwnProperty("display") ? t.display : null, e.style.display = t == null || typeof t == "boolean" ? "" : ("" + t).trim();
    }
    function _d(e, t) {
      e.nodeValue = t;
    }
    function Go(e) {
      var t = e.firstChild;
      for (t && t.nodeType === 10 && (t = t.nextSibling); t; ) {
        var a = t;
        switch (t = t.nextSibling, a.nodeName) {
          case "HTML":
          case "HEAD":
          case "BODY":
            Go(a), un(a);
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
    function zi(e, t, a, i) {
      for (; e.nodeType === 1; ) {
        var o = a;
        if (e.nodeName.toLowerCase() !== t.toLowerCase()) {
          if (!i && (e.nodeName !== "INPUT" || e.type !== "hidden"))
            break;
        } else if (i) {
          if (!e[ef])
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
        if (e = wl(e.nextSibling), e === null) break;
      }
      return null;
    }
    function Nl(e, t, a) {
      if (t === "") return null;
      for (; e.nodeType !== 3; )
        if ((e.nodeType !== 1 || e.nodeName !== "INPUT" || e.type !== "hidden") && !a || (e = wl(e.nextSibling), e === null)) return null;
      return e;
    }
    function au(e) {
      return e.data === pp || e.data === kc && e.ownerDocument.readyState === Tb;
    }
    function Lo(e, t) {
      var a = e.ownerDocument;
      if (e.data !== kc || a.readyState === Tb)
        t();
      else {
        var i = function() {
          t(), a.removeEventListener("DOMContentLoaded", i);
        };
        a.addEventListener("DOMContentLoaded", i), e._reactRetry = i;
      }
    }
    function wl(e) {
      for (; e != null; e = e.nextSibling) {
        var t = e.nodeType;
        if (t === 1 || t === 3) break;
        if (t === 8) {
          if (t = e.data, t === og || t === pp || t === kc || t === H0 || t === Sb)
            break;
          if (t === fg) return null;
        }
      }
      return e;
    }
    function Ud(e) {
      if (e.nodeType === 1) {
        for (var t = e.nodeName.toLowerCase(), a = {}, i = e.attributes, o = 0; o < i.length; o++) {
          var f = i[o];
          a[lm(f.name)] = f.name.toLowerCase() === "style" ? _c(e) : f.value;
        }
        return { type: t, props: a };
      }
      return e.nodeType === 8 ? { type: "Suspense", props: {} } : e.nodeValue;
    }
    function Cd(e, t, a) {
      return a === null || a[uT] !== !0 ? (e.nodeValue === t ? e = null : (t = Hl(t), e = Hl(e.nodeValue) === t ? null : e.nodeValue), e) : null;
    }
    function fm(e) {
      e = e.nextSibling;
      for (var t = 0; e; ) {
        if (e.nodeType === 8) {
          var a = e.data;
          if (a === fg) {
            if (t === 0)
              return wl(e.nextSibling);
            t--;
          } else
            a !== og && a !== pp && a !== kc || t++;
        }
        e = e.nextSibling;
      }
      return null;
    }
    function Vo(e) {
      e = e.previousSibling;
      for (var t = 0; e; ) {
        if (e.nodeType === 8) {
          var a = e.data;
          if (a === og || a === pp || a === kc) {
            if (t === 0) return e;
            t--;
          } else a === fg && t++;
        }
        e = e.previousSibling;
      }
      return null;
    }
    function rm(e) {
      Nc(e);
    }
    function Ua(e) {
      Nc(e);
    }
    function sm(e, t, a, i, o) {
      switch (o && ps(e, i.ancestorInfo), t = lt(a), e) {
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
    function Ca(e, t, a, i) {
      if (!a[qi] && Ml(a)) {
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
      $t(a, e, t), a[Kl] = i, a[ba] = t;
    }
    function Xo(e) {
      for (var t = e.attributes; t.length; )
        e.removeAttributeNode(t[0]);
      un(e);
    }
    function br(e) {
      return typeof e.getRootNode == "function" ? e.getRootNode() : e.nodeType === 9 ? e : e.ownerDocument;
    }
    function hv(e, t, a) {
      var i = Uh;
      if (i && typeof t == "string" && t) {
        var o = Aa(t);
        o = 'link[rel="' + e + '"][href="' + o + '"]', typeof a == "string" && (o += '[crossorigin="' + a + '"]'), zb.has(o) || (zb.add(o), e = { rel: e, crossOrigin: a, href: t }, i.querySelector(o) === null && (t = i.createElement("link"), $t(t, "link", e), z(t), i.head.appendChild(t)));
      }
    }
    function Gu(e, t, a, i) {
      var o = (o = uu.current) ? br(o) : null;
      if (!o)
        throw Error(
          '"resourceRoot" was expected to exist. This is a bug in React.'
        );
      switch (e) {
        case "meta":
        case "title":
          return null;
        case "style":
          return typeof a.precedence == "string" && typeof a.href == "string" ? (a = Mi(a.href), t = m(o).hoistableStyles, i = t.get(a), i || (i = {
            type: "style",
            instance: null,
            count: 0,
            state: null
          }, t.set(a, i)), i) : { type: "void", instance: null, count: 0, state: null };
        case "link":
          if (a.rel === "stylesheet" && typeof a.href == "string" && typeof a.precedence == "string") {
            e = Mi(a.href);
            var f = m(o).hoistableStyles, d = f.get(e);
            if (!d && (o = o.ownerDocument || o, d = {
              type: "stylesheet",
              instance: null,
              count: 0,
              state: { loading: ns, preload: null }
            }, f.set(e, d), (f = o.querySelector(
              nu(e)
            )) && !f._p && (d.instance = f, d.state.loading = vp | gu), !bu.has(e))) {
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
              bu.set(e, h), f || yv(
                o,
                e,
                h,
                d.state
              );
            }
            if (t && i === null)
              throw a = `

  - ` + Cc(t) + `
  + ` + Cc(a), Error(
                "Expected <link> not to update to be updated to a stylesheet with precedence. Check the `rel`, `href`, and `precedence` props of this component. Alternatively, check whether two different <link> components render in the same slot or share the same key." + a
              );
            return d;
          }
          if (t && i !== null)
            throw a = `

  - ` + Cc(t) + `
  + ` + Cc(a), Error(
              "Expected stylesheet with precedence to not be updated to a different kind of <link>. Check the `rel`, `href`, and `precedence` props of this component. Alternatively, check whether two different <link> components render in the same slot or share the same key." + a
            );
          return null;
        case "script":
          return t = a.async, a = a.src, typeof a == "string" && t && typeof t != "function" && typeof t != "symbol" ? (a = xc(a), t = m(o).hoistableScripts, i = t.get(a), i || (i = {
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
    function Cc(e) {
      var t = 0, a = "<link";
      return typeof e.rel == "string" ? (t++, a += ' rel="' + e.rel + '"') : Xu.call(e, "rel") && (t++, a += ' rel="' + (e.rel === null ? "null" : "invalid type " + typeof e.rel) + '"'), typeof e.href == "string" ? (t++, a += ' href="' + e.href + '"') : Xu.call(e, "href") && (t++, a += ' href="' + (e.href === null ? "null" : "invalid type " + typeof e.href) + '"'), typeof e.precedence == "string" ? (t++, a += ' precedence="' + e.precedence + '"') : Xu.call(e, "precedence") && (t++, a += " precedence={" + (e.precedence === null ? "null" : "invalid type " + typeof e.precedence) + "}"), Object.getOwnPropertyNames(e).length > t && (a += " ..."), a + " />";
    }
    function Mi(e) {
      return 'href="' + Aa(e) + '"';
    }
    function nu(e) {
      return 'link[rel="stylesheet"][' + e + "]";
    }
    function dm(e) {
      return ke({}, e, {
        "data-precedence": e.precedence,
        precedence: null
      });
    }
    function yv(e, t, a, i) {
      e.querySelector(
        'link[rel="preload"][as="style"][' + t + "]"
      ) ? i.loading = vp : (t = e.createElement("link"), i.preload = t, t.addEventListener("load", function() {
        return i.loading |= vp;
      }), t.addEventListener("error", function() {
        return i.loading |= Ob;
      }), $t(t, "link", a), z(t), e.head.appendChild(t));
    }
    function xc(e) {
      return '[src="' + Aa(e) + '"]';
    }
    function Hc(e) {
      return "script[async]" + e;
    }
    function xd(e, t, a) {
      if (t.count++, t.instance === null)
        switch (t.type) {
          case "style":
            var i = e.querySelector(
              'style[data-href~="' + Aa(a.href) + '"]'
            );
            if (i)
              return t.instance = i, z(i), i;
            var o = ke({}, a, {
              "data-href": a.href,
              "data-precedence": a.precedence,
              href: null,
              precedence: null
            });
            return i = (e.ownerDocument || e).createElement("style"), z(i), $t(i, "style", o), Hd(i, a.precedence, e), t.instance = i;
          case "stylesheet":
            o = Mi(a.href);
            var f = e.querySelector(
              nu(o)
            );
            if (f)
              return t.state.loading |= gu, t.instance = f, z(f), f;
            i = dm(a), (o = bu.get(o)) && hm(i, o), f = (e.ownerDocument || e).createElement("link"), z(f);
            var d = f;
            return d._p = new Promise(function(h, v) {
              d.onload = h, d.onerror = v;
            }), $t(f, "link", i), t.state.loading |= gu, Hd(f, a.precedence, e), t.instance = f;
          case "script":
            return f = xc(a.src), (o = e.querySelector(
              Hc(f)
            )) ? (t.instance = o, z(o), o) : (i = a, (o = bu.get(f)) && (i = ke({}, a), ym(i, o)), e = e.ownerDocument || e, o = e.createElement("script"), z(o), $t(o, "link", i), e.head.appendChild(o), t.instance = o);
          case "void":
            return null;
          default:
            throw Error(
              'acquireResource encountered a resource type it did not expect: "' + t.type + '". this is a bug in React.'
            );
        }
      else
        t.type === "stylesheet" && (t.state.loading & gu) === ns && (i = t.instance, t.state.loading |= gu, Hd(i, a.precedence, e));
      return t.instance;
    }
    function Hd(e, t, a) {
      for (var i = a.querySelectorAll(
        'link[rel="stylesheet"][data-precedence],style[data-precedence]'
      ), o = i.length ? i[i.length - 1] : null, f = o, d = 0; d < i.length; d++) {
        var h = i[d];
        if (h.dataset.precedence === t) f = h;
        else if (f !== o) break;
      }
      f ? f.parentNode.insertBefore(e, f.nextSibling) : (t = a.nodeType === 9 ? a.head : a, t.insertBefore(e, t.firstChild));
    }
    function hm(e, t) {
      e.crossOrigin == null && (e.crossOrigin = t.crossOrigin), e.referrerPolicy == null && (e.referrerPolicy = t.referrerPolicy), e.title == null && (e.title = t.title);
    }
    function ym(e, t) {
      e.crossOrigin == null && (e.crossOrigin = t.crossOrigin), e.referrerPolicy == null && (e.referrerPolicy = t.referrerPolicy), e.integrity == null && (e.integrity = t.integrity);
    }
    function mm(e, t, a) {
      if (sg === null) {
        var i = /* @__PURE__ */ new Map(), o = sg = /* @__PURE__ */ new Map();
        o.set(a, i);
      } else
        o = sg, i = o.get(a), i || (i = /* @__PURE__ */ new Map(), o.set(a, i));
      if (i.has(e)) return i;
      for (i.set(e, null), a = a.getElementsByTagName(e), o = 0; o < a.length; o++) {
        var f = a[o];
        if (!(f[ef] || f[Kl] || e === "link" && f.getAttribute("rel") === "stylesheet") && f.namespaceURI !== nf) {
          var d = f.getAttribute(t) || "";
          d = e + d;
          var h = i.get(d);
          h ? h.push(f) : i.set(d, [f]);
        }
      }
      return i;
    }
    function pm(e, t, a) {
      e = e.ownerDocument || e, e.head.insertBefore(
        a,
        t === "title" ? e.querySelector("head > title") : null
      );
    }
    function Qo(e, t, a) {
      var i = !a.ancestorInfo.containerTagInScope;
      if (a.context === _h || t.itemProp != null)
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
    function Sr(e) {
      return !(e.type === "stylesheet" && (e.state.loading & Db) === ns);
    }
    function mv() {
    }
    function pv(e, t, a) {
      if (gp === null)
        throw Error(
          "Internal React Error: suspendedState null when it was expected to exists. Please report this as a React bug."
        );
      var i = gp;
      if (t.type === "stylesheet" && (typeof a.media != "string" || matchMedia(a.media).matches !== !1) && (t.state.loading & gu) === ns) {
        if (t.instance === null) {
          var o = Mi(a.href), f = e.querySelector(
            nu(o)
          );
          if (f) {
            e = f._p, e !== null && typeof e == "object" && typeof e.then == "function" && (i.count++, i = Tr.bind(i), e.then(i, i)), t.state.loading |= gu, t.instance = f, z(f);
            return;
          }
          f = e.ownerDocument || e, a = dm(a), (o = bu.get(o)) && hm(a, o), f = f.createElement("link"), z(f);
          var d = f;
          d._p = new Promise(function(h, v) {
            d.onload = h, d.onerror = v;
          }), $t(f, "link", a), t.instance = f;
        }
        i.stylesheets === null && (i.stylesheets = /* @__PURE__ */ new Map()), i.stylesheets.set(t, e), (e = t.state.preload) && (t.state.loading & Db) === ns && (i.count++, t = Tr.bind(i), e.addEventListener("load", t), e.addEventListener("error", t));
      }
    }
    function vv() {
      if (gp === null)
        throw Error(
          "Internal React Error: suspendedState null when it was expected to exists. Please report this as a React bug."
        );
      var e = gp;
      return e.stylesheets && e.count === 0 && Nd(e, e.stylesheets), 0 < e.count ? function(t) {
        var a = setTimeout(function() {
          if (e.stylesheets && Nd(e, e.stylesheets), e.unsuspend) {
            var i = e.unsuspend;
            e.unsuspend = null, i();
          }
        }, 6e4);
        return e.unsuspend = t, function() {
          e.unsuspend = null, clearTimeout(a);
        };
      } : null;
    }
    function Tr() {
      if (this.count--, this.count === 0) {
        if (this.stylesheets)
          Nd(this, this.stylesheets);
        else if (this.unsuspend) {
          var e = this.unsuspend;
          this.unsuspend = null, e();
        }
      }
    }
    function Nd(e, t) {
      e.stylesheets = null, e.unsuspend !== null && (e.count++, dg = /* @__PURE__ */ new Map(), t.forEach(gv, e), dg = null, Tr.call(e));
    }
    function gv(e, t) {
      if (!(t.state.loading & gu)) {
        var a = dg.get(e);
        if (a) var i = a.get(Y0);
        else {
          a = /* @__PURE__ */ new Map(), dg.set(e, a);
          for (var o = e.querySelectorAll(
            "link[data-precedence],style[data-precedence]"
          ), f = 0; f < o.length; f++) {
            var d = o[f];
            (d.nodeName === "LINK" || d.getAttribute("media") !== "not all") && (a.set(d.dataset.precedence, d), i = d);
          }
          i && a.set(Y0, i);
        }
        o = t.instance, d = o.getAttribute("data-precedence"), f = a.get(d) || i, f === i && a.set(Y0, o), a.set(d, o), this.count++, i = Tr.bind(this), o.addEventListener("load", i), o.addEventListener("error", i), f ? f.parentNode.insertBefore(o, f.nextSibling) : (e = e.nodeType === 9 ? e.head : e, e.insertBefore(o, e.firstChild)), t.state.loading |= gu;
      }
    }
    function wd(e, t, a, i, o, f, d, h) {
      for (this.tag = 1, this.containerInfo = e, this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = as, this.callbackNode = this.next = this.pendingContext = this.context = this.cancelPendingCommit = null, this.callbackPriority = 0, this.expirationTimes = to(-1), this.entangledLanes = this.shellSuspendCounter = this.errorRecoveryDisabledLanes = this.expiredLanes = this.warmLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = to(0), this.hiddenUpdates = to(null), this.identifierPrefix = i, this.onUncaughtError = o, this.onCaughtError = f, this.onRecoverableError = d, this.pooledCache = null, this.pooledCacheLanes = 0, this.formState = h, this.incompleteTransitions = /* @__PURE__ */ new Map(), this.passiveEffectDuration = this.effectDuration = -0, this.memoizedUpdaters = /* @__PURE__ */ new Set(), e = this.pendingUpdatersLaneMap = [], t = 0; 31 > t; t++) e.push(/* @__PURE__ */ new Set());
      this._debugRootType = a ? "hydrateRoot()" : "createRoot()";
    }
    function vm(e, t, a, i, o, f, d, h, v, b, q, L) {
      return e = new wd(
        e,
        t,
        a,
        d,
        h,
        v,
        b,
        L
      ), t = wS, f === !0 && (t |= Ta | ku), It && (t |= la), f = D(3, null, null, t), e.current = f, f.stateNode = e, t = jf(), oc(t), e.pooledCache = t, oc(t), f.memoizedState = {
        element: i,
        isDehydrated: a,
        cache: t
      }, oa(f), e;
    }
    function gm(e) {
      return e ? (e = uf, e) : uf;
    }
    function Tt(e, t, a, i, o, f) {
      if (ql && typeof ql.onScheduleFiberRoot == "function")
        try {
          ql.onScheduleFiberRoot(Ni, i, a);
        } catch (d) {
          ga || (ga = !0, console.error(
            "React instrumentation encountered an error: %s",
            d
          ));
        }
      fe !== null && typeof fe.markRenderScheduled == "function" && fe.markRenderScheduled(t), o = gm(o), i.context === null ? i.context = o : i.pendingContext = o, Sa && Ha !== null && !Cb && (Cb = !0, console.error(
        `Render methods should be a pure function of props and state; triggering nested component updates from render is not allowed. If necessary, trigger nested updates in componentDidUpdate.

Check the render method of %s.`,
        de(Ha) || "Unknown"
      )), i = Bn(t), i.payload = { element: a }, f = f === void 0 ? null : f, f !== null && (typeof f != "function" && console.error(
        "Expected the last optional `callback` argument to be a function. Instead received: %s.",
        f
      ), i.callback = f), a = yn(e, i, t), a !== null && (Jt(a, e, t), hi(a, e, t));
    }
    function qd(e, t) {
      if (e = e.memoizedState, e !== null && e.dehydrated !== null) {
        var a = e.retryLane;
        e.retryLane = a !== 0 && a < t ? a : t;
      }
    }
    function bm(e, t) {
      qd(e, t), (e = e.alternate) && qd(e, t);
    }
    function Sm(e) {
      if (e.tag === 13) {
        var t = ca(e, 67108864);
        t !== null && Jt(t, e, 67108864), bm(e, 67108864);
      }
    }
    function xg() {
      return Ha;
    }
    function Hg() {
      for (var e = /* @__PURE__ */ new Map(), t = 1, a = 0; 31 > a; a++) {
        var i = Sf(t);
        e.set(t, i), t *= 2;
      }
      return e;
    }
    function Ng(e, t, a, i) {
      var o = Y.T;
      Y.T = null;
      var f = Ue.p;
      try {
        Ue.p = Bl, _i(e, t, a, i);
      } finally {
        Ue.p = f, Y.T = o;
      }
    }
    function Bd(e, t, a, i) {
      var o = Y.T;
      Y.T = null;
      var f = Ue.p;
      try {
        Ue.p = Rn, _i(e, t, a, i);
      } finally {
        Ue.p = f, Y.T = o;
      }
    }
    function _i(e, t, a, i) {
      if (yg) {
        var o = Er(i);
        if (o === null)
          Il(
            e,
            t,
            i,
            mg,
            a
          ), Ui(e, i);
        else if (Rr(
          o,
          e,
          t,
          a,
          i
        ))
          i.stopPropagation();
        else if (Ui(e, i), t & 4 && -1 < hT.indexOf(e)) {
          for (; o !== null; ) {
            var f = Ml(o);
            if (f !== null)
              switch (f.tag) {
                case 3:
                  if (f = f.stateNode, f.current.memoizedState.isDehydrated) {
                    var d = ll(f.pendingLanes);
                    if (d !== 0) {
                      var h = f;
                      for (h.pendingLanes |= 2, h.entangledLanes |= 2; d; ) {
                        var v = 1 << 31 - Zl(d);
                        h.entanglements[1] |= v, d &= ~v;
                      }
                      Wa(f), (Et & (Ba | Iu)) === An && (Iv = iu() + fb, zc(0));
                    }
                  }
                  break;
                case 13:
                  h = ca(f, 2), h !== null && Jt(h, f, 2), Oc(), bm(f, 2);
              }
            if (f = Er(i), f === null && Il(
              e,
              t,
              i,
              mg,
              a
            ), f === o) break;
            o = f;
          }
          o !== null && i.stopPropagation();
        } else
          Il(
            e,
            t,
            i,
            null,
            a
          );
      }
    }
    function Er(e) {
      return e = ec(e), Zo(e);
    }
    function Zo(e) {
      if (mg = null, e = ia(e), e !== null) {
        var t = Pe(e);
        if (t === null) e = null;
        else {
          var a = t.tag;
          if (a === 13) {
            if (e = Ct(t), e !== null) return e;
            e = null;
          } else if (a === 3) {
            if (t.stateNode.current.memoizedState.isDehydrated)
              return t.tag === 3 ? t.stateNode.containerInfo : null;
            e = null;
          } else t !== e && (e = null);
        }
      }
      return mg = e, null;
    }
    function Yd(e) {
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
          return Bl;
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
          return Rn;
        case "message":
          switch (Hi()) {
            case Xd:
              return Bl;
            case xr:
              return Rn;
            case Fo:
            case jg:
              return Zu;
            case Hr:
              return Jd;
            default:
              return Zu;
          }
        default:
          return Zu;
      }
    }
    function Ui(e, t) {
      switch (e) {
        case "focusin":
        case "focusout":
          pf = null;
          break;
        case "dragenter":
        case "dragleave":
          vf = null;
          break;
        case "mouseover":
        case "mouseout":
          gf = null;
          break;
        case "pointerover":
        case "pointerout":
          Sp.delete(t.pointerId);
          break;
        case "gotpointercapture":
        case "lostpointercapture":
          Tp.delete(t.pointerId);
      }
    }
    function va(e, t, a, i, o, f) {
      return e === null || e.nativeEvent !== f ? (e = {
        blockedOn: t,
        domEventName: a,
        eventSystemFlags: i,
        nativeEvent: f,
        targetContainers: [o]
      }, t !== null && (t = Ml(t), t !== null && Sm(t)), e) : (e.eventSystemFlags |= i, t = e.targetContainers, o !== null && t.indexOf(o) === -1 && t.push(o), e);
    }
    function Rr(e, t, a, i, o) {
      switch (t) {
        case "focusin":
          return pf = va(
            pf,
            e,
            t,
            a,
            i,
            o
          ), !0;
        case "dragenter":
          return vf = va(
            vf,
            e,
            t,
            a,
            i,
            o
          ), !0;
        case "mouseover":
          return gf = va(
            gf,
            e,
            t,
            a,
            i,
            o
          ), !0;
        case "pointerover":
          var f = o.pointerId;
          return Sp.set(
            f,
            va(
              Sp.get(f) || null,
              e,
              t,
              a,
              i,
              o
            )
          ), !0;
        case "gotpointercapture":
          return f = o.pointerId, Tp.set(
            f,
            va(
              Tp.get(f) || null,
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
    function bv(e) {
      var t = ia(e.target);
      if (t !== null) {
        var a = Pe(t);
        if (a !== null) {
          if (t = a.tag, t === 13) {
            if (t = Ct(a), t !== null) {
              e.blockedOn = t, ao(e.priority, function() {
                if (a.tag === 13) {
                  var i = ya(a);
                  i = Dl(i);
                  var o = ca(
                    a,
                    i
                  );
                  o !== null && Jt(o, a, i), bm(a, i);
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
    function Ar(e) {
      if (e.blockedOn !== null) return !1;
      for (var t = e.targetContainers; 0 < t.length; ) {
        var a = Er(e.nativeEvent);
        if (a === null) {
          a = e.nativeEvent;
          var i = new a.constructor(
            a.type,
            a
          ), o = i;
          r !== null && console.error(
            "Expected currently replaying event to be null. This error is likely caused by a bug in React. Please file an issue."
          ), r = o, a.target.dispatchEvent(i), r === null && console.error(
            "Expected currently replaying event to not be null. This error is likely caused by a bug in React. Please file an issue."
          ), r = null;
        } else
          return t = Ml(a), t !== null && Sm(t), e.blockedOn = a, !1;
        t.shift();
      }
      return !0;
    }
    function Tm(e, t, a) {
      Ar(e) && a.delete(t);
    }
    function Sv() {
      j0 = !1, pf !== null && Ar(pf) && (pf = null), vf !== null && Ar(vf) && (vf = null), gf !== null && Ar(gf) && (gf = null), Sp.forEach(Tm), Tp.forEach(Tm);
    }
    function Or(e, t) {
      e.blockedOn === t && (e.blockedOn = null, j0 || (j0 = !0, Ft.unstable_scheduleCallback(
        Ft.unstable_NormalPriority,
        Sv
      )));
    }
    function Tv(e) {
      pg !== e && (pg = e, Ft.unstable_scheduleCallback(
        Ft.unstable_NormalPriority,
        function() {
          pg === e && (pg = null);
          for (var t = 0; t < e.length; t += 3) {
            var a = e[t], i = e[t + 1], o = e[t + 2];
            if (typeof i != "function") {
              if (Zo(i || a) === null)
                continue;
              break;
            }
            var f = Ml(a);
            f !== null && (e.splice(t, 3), t -= 3, a = {
              pending: !0,
              data: o,
              method: a.method,
              action: i
            }, Object.freeze(a), yc(
              f,
              a,
              i,
              o
            ));
          }
        }
      ));
    }
    function Nc(e) {
      function t(v) {
        return Or(v, e);
      }
      pf !== null && Or(pf, e), vf !== null && Or(vf, e), gf !== null && Or(gf, e), Sp.forEach(t), Tp.forEach(t);
      for (var a = 0; a < bf.length; a++) {
        var i = bf[a];
        i.blockedOn === e && (i.blockedOn = null);
      }
      for (; 0 < bf.length && (a = bf[0], a.blockedOn === null); )
        bv(a), a.blockedOn === null && bf.shift();
      if (a = (e.ownerDocument || e).$$reactFormReplay, a != null)
        for (i = 0; i < a.length; i += 3) {
          var o = a[i], f = a[i + 1], d = o[ba] || null;
          if (typeof f == "function")
            d || Tv(a);
          else if (d) {
            var h = null;
            if (f && f.hasAttribute("formAction")) {
              if (o = f, d = f[ba] || null)
                h = d.formAction;
              else if (Zo(o) !== null) continue;
            } else h = d.action;
            typeof h == "function" ? a[i + 1] = h : (a.splice(i, 3), i -= 3), Tv(a);
          }
        }
    }
    function jd(e) {
      this._internalRoot = e;
    }
    function Dr(e) {
      this._internalRoot = e;
    }
    function Ev(e) {
      e[qi] && (e._reactRootContainer ? console.error(
        "You are calling ReactDOMClient.createRoot() on a container that was previously passed to ReactDOM.render(). This is not supported."
      ) : console.error(
        "You are calling ReactDOMClient.createRoot() on a container that has already been passed to createRoot() before. Instead, call root.render() on the existing root instead if you want to update it."
      ));
    }
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var Ft = cS(), zr = xh(), wg = oS(), ke = Object.assign, Mr = Symbol.for("react.element"), Ci = Symbol.for("react.transitional.element"), wc = Symbol.for("react.portal"), Ve = Symbol.for("react.fragment"), Ko = Symbol.for("react.strict_mode"), Jo = Symbol.for("react.profiler"), Em = Symbol.for("react.provider"), Gd = Symbol.for("react.consumer"), Pa = Symbol.for("react.context"), Lu = Symbol.for("react.forward_ref"), ko = Symbol.for("react.suspense"), xi = Symbol.for("react.suspense_list"), _r = Symbol.for("react.memo"), xa = Symbol.for("react.lazy"), Rm = Symbol.for("react.activity"), Rv = Symbol.for("react.memo_cache_sentinel"), Am = Symbol.iterator, Ld = Symbol.for("react.client.reference"), qe = Array.isArray, Y = zr.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, Ue = wg.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, qg = Object.freeze({
      pending: !1,
      data: null,
      method: null,
      action: null
    }), Ur = [], Cr = [], en = -1, Vu = Ot(null), $o = Ot(null), uu = Ot(null), Wo = Ot(null), Xu = Object.prototype.hasOwnProperty, Vd = Ft.unstable_scheduleCallback, Bg = Ft.unstable_cancelCallback, Av = Ft.unstable_shouldYield, Yg = Ft.unstable_requestPaint, iu = Ft.unstable_now, Hi = Ft.unstable_getCurrentPriorityLevel, Xd = Ft.unstable_ImmediatePriority, xr = Ft.unstable_UserBlockingPriority, Fo = Ft.unstable_NormalPriority, jg = Ft.unstable_LowPriority, Hr = Ft.unstable_IdlePriority, Gg = Ft.log, En = Ft.unstable_setDisableYieldValue, Ni = null, ql = null, fe = null, ga = !1, It = typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u", Zl = Math.clz32 ? Math.clz32 : eo, Qd = Math.log, Qu = Math.LN2, Zd = 256, Kd = 4194304, Bl = 2, Rn = 8, Zu = 32, Jd = 268435456, wi = Math.random().toString(36).slice(2), Kl = "__reactFiber$" + wi, ba = "__reactProps$" + wi, qi = "__reactContainer$" + wi, Om = "__reactEvents$" + wi, Ov = "__reactListeners$" + wi, Io = "__reactHandles$" + wi, Po = "__reactResources$" + wi, ef = "__reactMarker$" + wi, Dv = /* @__PURE__ */ new Set(), tn = {}, qc = {}, zv = {
      button: !0,
      checkbox: !0,
      image: !0,
      hidden: !0,
      radio: !0,
      reset: !0,
      submit: !0
    }, kd = RegExp(
      "^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), $d = {}, Wd = {}, Bi = 0, Dm, zm, Mv, Mm, tf, _v, Uv;
    on.__reactDisabledLog = !0;
    var _m, Nr, lf = !1, wr = new (typeof WeakMap == "function" ? WeakMap : Map)(), Ha = null, Sa = !1, Lg = /[\n"\\]/g, Um = !1, Cm = !1, xm = !1, Hm = !1, Fd = !1, Nm = !1, qr = ["value", "defaultValue"], Cv = !1, xv = /["'&<>\n\t]|^\s|\s$/, wm = "address applet area article aside base basefont bgsound blockquote body br button caption center col colgroup dd details dir div dl dt embed fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hgroup hr html iframe img input isindex li link listing main marquee menu menuitem meta nav noembed noframes noscript object ol p param plaintext pre script section select source style summary table tbody td template textarea tfoot th thead title tr track ul wbr xmp".split(
      " "
    ), Id = "applet caption html table td th marquee object template foreignObject desc title".split(
      " "
    ), Pd = Id.concat(["button"]), qm = "dd dt li option optgroup p rp rt".split(" "), Bm = {
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
    }, af = {}, cu = {
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
    }, Ku = /([A-Z])/g, Ju = /^ms-/, Br = /^(?:webkit|moz|o)[A-Z]/, Yr = /^-ms-/, Yi = /-(.)/g, Hv = /;\s*$/, Bc = {}, Yc = {}, Nv = !1, Ym = !1, jr = new Set(
      "animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(
        " "
      )
    ), Gr = "http://www.w3.org/1998/Math/MathML", nf = "http://www.w3.org/2000/svg", eh = /* @__PURE__ */ new Map([
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
    ]), jc = {
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
    }, jm = {
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
    }, ou = {}, Gm = RegExp(
      "^(aria)-[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), th = RegExp(
      "^(aria)[A-Z][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), Lm = !1, ta = {}, Lr = /^on./, l = /^on[^A-Z]/, n = RegExp(
      "^(aria)-[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), u = RegExp(
      "^(aria)[A-Z][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), c = /^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i, r = null, s = null, y = null, p = !1, S = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), C = !1;
    if (S)
      try {
        var Z = {};
        Object.defineProperty(Z, "passive", {
          get: function() {
            C = !0;
          }
        }), window.addEventListener("test", Z, Z), window.removeEventListener("test", Z, Z);
      } catch {
        C = !1;
      }
    var W = null, N = null, B = null, Te = {
      eventPhase: 0,
      bubbles: 0,
      cancelable: 0,
      timeStamp: function(e) {
        return e.timeStamp || Date.now();
      },
      defaultPrevented: 0,
      isTrusted: 0
    }, Ee = Ul(Te), yt = ke({}, Te, { view: 0, detail: 0 }), M = Ul(yt), O, U, $, se = ke({}, yt, {
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
      getModifierState: Ss,
      button: 0,
      buttons: 0,
      relatedTarget: function(e) {
        return e.relatedTarget === void 0 ? e.fromElement === e.srcElement ? e.toElement : e.fromElement : e.relatedTarget;
      },
      movementX: function(e) {
        return "movementX" in e ? e.movementX : (e !== $ && ($ && e.type === "mousemove" ? (O = e.screenX - $.screenX, U = e.screenY - $.screenY) : U = O = 0, $ = e), O);
      },
      movementY: function(e) {
        return "movementY" in e ? e.movementY : U;
      }
    }), We = Ul(se), Se = ke({}, se, { dataTransfer: 0 }), _e = Ul(Se), Rl = ke({}, yt, { relatedTarget: 0 }), ft = Ul(Rl), ji = ke({}, Te, {
      animationName: 0,
      elapsedTime: 0,
      pseudoElement: 0
    }), Vg = Ul(ji), rS = ke({}, Te, {
      clipboardData: function(e) {
        return "clipboardData" in e ? e.clipboardData : window.clipboardData;
      }
    }), sS = Ul(rS), dS = ke({}, Te, { data: 0 }), Q0 = Ul(
      dS
    ), hS = Q0, yS = {
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
    }, mS = {
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
    }, pS = {
      Alt: "altKey",
      Control: "ctrlKey",
      Meta: "metaKey",
      Shift: "shiftKey"
    }, vS = ke({}, yt, {
      key: function(e) {
        if (e.key) {
          var t = yS[e.key] || e.key;
          if (t !== "Unidentified") return t;
        }
        return e.type === "keypress" ? (e = ro(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? mS[e.keyCode] || "Unidentified" : "";
      },
      code: 0,
      location: 0,
      ctrlKey: 0,
      shiftKey: 0,
      altKey: 0,
      metaKey: 0,
      repeat: 0,
      locale: 0,
      getModifierState: Ss,
      charCode: function(e) {
        return e.type === "keypress" ? ro(e) : 0;
      },
      keyCode: function(e) {
        return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
      },
      which: function(e) {
        return e.type === "keypress" ? ro(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
      }
    }), gS = Ul(vS), bS = ke({}, se, {
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
    }), Z0 = Ul(bS), SS = ke({}, yt, {
      touches: 0,
      targetTouches: 0,
      changedTouches: 0,
      altKey: 0,
      metaKey: 0,
      ctrlKey: 0,
      shiftKey: 0,
      getModifierState: Ss
    }), TS = Ul(SS), ES = ke({}, Te, {
      propertyName: 0,
      elapsedTime: 0,
      pseudoElement: 0
    }), RS = Ul(ES), AS = ke({}, se, {
      deltaX: function(e) {
        return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
      },
      deltaY: function(e) {
        return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
      },
      deltaZ: 0,
      deltaMode: 0
    }), OS = Ul(AS), DS = ke({}, Te, {
      newState: 0,
      oldState: 0
    }), zS = Ul(DS), MS = [9, 13, 27, 32], K0 = 229, Xg = S && "CompositionEvent" in window, Vm = null;
    S && "documentMode" in document && (Vm = document.documentMode);
    var _S = S && "TextEvent" in window && !Vm, J0 = S && (!Xg || Vm && 8 < Vm && 11 >= Vm), k0 = 32, $0 = String.fromCharCode(k0), W0 = !1, lh = !1, US = {
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
    }, Xm = null, Qm = null, F0 = !1;
    S && (F0 = Vh("input") && (!document.documentMode || 9 < document.documentMode));
    var Na = typeof Object.is == "function" ? Object.is : Mg, CS = S && "documentMode" in document && 11 >= document.documentMode, ah = null, Qg = null, Zm = null, Zg = !1, nh = {
      animationend: Ou("Animation", "AnimationEnd"),
      animationiteration: Ou("Animation", "AnimationIteration"),
      animationstart: Ou("Animation", "AnimationStart"),
      transitionrun: Ou("Transition", "TransitionRun"),
      transitionstart: Ou("Transition", "TransitionStart"),
      transitioncancel: Ou("Transition", "TransitionCancel"),
      transitionend: Ou("Transition", "TransitionEnd")
    }, Kg = {}, I0 = {};
    S && (I0 = document.createElement("div").style, "AnimationEvent" in window || (delete nh.animationend.animation, delete nh.animationiteration.animation, delete nh.animationstart.animation), "TransitionEvent" in window || delete nh.transitionend.transition);
    var P0 = ac("animationend"), e1 = ac("animationiteration"), t1 = ac("animationstart"), xS = ac("transitionrun"), HS = ac("transitionstart"), NS = ac("transitioncancel"), l1 = ac("transitionend"), a1 = /* @__PURE__ */ new Map(), Jg = "abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(
      " "
    );
    Jg.push("scrollEnd");
    var kg = /* @__PURE__ */ new WeakMap(), wv = 1, Gc = 2, fu = [], uh = 0, $g = 0, uf = {};
    Object.freeze(uf);
    var ru = null, ih = null, Gt = 0, wS = 1, la = 2, Ta = 8, ku = 16, n1 = 64, u1 = !1;
    try {
      var i1 = Object.preventExtensions({});
    } catch {
      u1 = !0;
    }
    var ch = [], oh = 0, qv = null, Bv = 0, su = [], du = 0, Vr = null, Lc = 1, Vc = "", wa = null, ul = null, mt = !1, Xc = !1, hu = null, Xr = null, Gi = !1, Wg = Error(
      "Hydration Mismatch Exception: This is not a real error, and should not leak into userspace. If you're seeing this, it's likely a bug in React."
    ), c1 = 0;
    if (typeof performance == "object" && typeof performance.now == "function")
      var qS = performance, o1 = function() {
        return qS.now();
      };
    else {
      var BS = Date;
      o1 = function() {
        return BS.now();
      };
    }
    var Fg = Ot(null), Ig = Ot(null), f1 = {}, Yv = null, fh = null, rh = !1, YS = typeof AbortController < "u" ? AbortController : function() {
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
    }, jS = Ft.unstable_scheduleCallback, GS = Ft.unstable_NormalPriority, Yl = {
      $$typeof: Pa,
      Consumer: null,
      Provider: null,
      _currentValue: null,
      _currentValue2: null,
      _threadCount: 0,
      _currentRenderer: null,
      _currentRenderer2: null
    }, sh = Ft.unstable_now, r1 = -0, jv = -0, ln = -1.1, Qr = -0, Gv = !1, Lv = !1, Km = null, Pg = 0, Zr = 0, dh = null, s1 = Y.S;
    Y.S = function(e, t) {
      typeof t == "object" && t !== null && typeof t.then == "function" && Zp(e, t), s1 !== null && s1(e, t);
    };
    var Kr = Ot(null), $u = {
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
    }, Jm = [], km = [], $m = [], Wm = [], Fm = [], Im = [], Jr = /* @__PURE__ */ new Set();
    $u.recordUnsafeLifecycleWarnings = function(e, t) {
      Jr.has(e.type) || (typeof t.componentWillMount == "function" && t.componentWillMount.__suppressDeprecationWarning !== !0 && Jm.push(e), e.mode & Ta && typeof t.UNSAFE_componentWillMount == "function" && km.push(e), typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps.__suppressDeprecationWarning !== !0 && $m.push(e), e.mode & Ta && typeof t.UNSAFE_componentWillReceiveProps == "function" && Wm.push(e), typeof t.componentWillUpdate == "function" && t.componentWillUpdate.__suppressDeprecationWarning !== !0 && Fm.push(e), e.mode & Ta && typeof t.UNSAFE_componentWillUpdate == "function" && Im.push(e));
    }, $u.flushPendingUnsafeLifecycleWarnings = function() {
      var e = /* @__PURE__ */ new Set();
      0 < Jm.length && (Jm.forEach(function(h) {
        e.add(
          de(h) || "Component"
        ), Jr.add(h.type);
      }), Jm = []);
      var t = /* @__PURE__ */ new Set();
      0 < km.length && (km.forEach(function(h) {
        t.add(
          de(h) || "Component"
        ), Jr.add(h.type);
      }), km = []);
      var a = /* @__PURE__ */ new Set();
      0 < $m.length && ($m.forEach(function(h) {
        a.add(
          de(h) || "Component"
        ), Jr.add(h.type);
      }), $m = []);
      var i = /* @__PURE__ */ new Set();
      0 < Wm.length && (Wm.forEach(
        function(h) {
          i.add(
            de(h) || "Component"
          ), Jr.add(h.type);
        }
      ), Wm = []);
      var o = /* @__PURE__ */ new Set();
      0 < Fm.length && (Fm.forEach(function(h) {
        o.add(
          de(h) || "Component"
        ), Jr.add(h.type);
      }), Fm = []);
      var f = /* @__PURE__ */ new Set();
      if (0 < Im.length && (Im.forEach(function(h) {
        f.add(
          de(h) || "Component"
        ), Jr.add(h.type);
      }), Im = []), 0 < t.size) {
        var d = K(
          t
        );
        console.error(
          `Using UNSAFE_componentWillMount in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move code with side effects to componentDidMount, and set initial state in the constructor.

Please update the following components: %s`,
          d
        );
      }
      0 < i.size && (d = K(
        i
      ), console.error(
        `Using UNSAFE_componentWillReceiveProps in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* If you're updating state whenever props change, refactor your code to use memoization techniques or move it to static getDerivedStateFromProps. Learn more at: https://react.dev/link/derived-state

Please update the following components: %s`,
        d
      )), 0 < f.size && (d = K(
        f
      ), console.error(
        `Using UNSAFE_componentWillUpdate in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.

Please update the following components: %s`,
        d
      )), 0 < e.size && (d = K(e), console.warn(
        `componentWillMount has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move code with side effects to componentDidMount, and set initial state in the constructor.
* Rename componentWillMount to UNSAFE_componentWillMount to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      )), 0 < a.size && (d = K(
        a
      ), console.warn(
        `componentWillReceiveProps has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* If you're updating state whenever props change, refactor your code to use memoization techniques or move it to static getDerivedStateFromProps. Learn more at: https://react.dev/link/derived-state
* Rename componentWillReceiveProps to UNSAFE_componentWillReceiveProps to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      )), 0 < o.size && (d = K(o), console.warn(
        `componentWillUpdate has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* Rename componentWillUpdate to UNSAFE_componentWillUpdate to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      ));
    };
    var Vv = /* @__PURE__ */ new Map(), d1 = /* @__PURE__ */ new Set();
    $u.recordLegacyContextWarning = function(e, t) {
      for (var a = null, i = e; i !== null; )
        i.mode & Ta && (a = i), i = i.return;
      a === null ? console.error(
        "Expected to find a StrictMode component in a strict mode tree. This error is likely caused by a bug in React. Please file an issue."
      ) : !d1.has(e.type) && (i = Vv.get(a), e.type.contextTypes != null || e.type.childContextTypes != null || t !== null && typeof t.getChildContext == "function") && (i === void 0 && (i = [], Vv.set(a, i)), i.push(e));
    }, $u.flushLegacyContextWarning = function() {
      Vv.forEach(function(e) {
        if (e.length !== 0) {
          var t = e[0], a = /* @__PURE__ */ new Set();
          e.forEach(function(o) {
            a.add(de(o) || "Component"), d1.add(o.type);
          });
          var i = K(a);
          he(t, function() {
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
    }, $u.discardPendingWarnings = function() {
      Jm = [], km = [], $m = [], Wm = [], Fm = [], Im = [], Vv = /* @__PURE__ */ new Map();
    };
    var Pm = Error(
      "Suspense Exception: This is not a real error! It's an implementation detail of `use` to interrupt the current render. You must either rethrow it immediately, or move the `use` call outside of the `try/catch` block. Capturing without rethrowing will lead to unexpected behavior.\n\nTo handle async errors, wrap your component in an error boundary, or call the promise's `.catch` method and pass the result to `use`."
    ), h1 = Error(
      "Suspense Exception: This is not a real error, and should not leak into userspace. If you're seeing this, it's likely a bug in React."
    ), Xv = Error(
      "Suspense Exception: This is not a real error! It's an implementation detail of `useActionState` to interrupt the current render. You must either rethrow it immediately, or move the `useActionState` call outside of the `try/catch` block. Capturing without rethrowing will lead to unexpected behavior.\n\nTo handle async errors, wrap your component in an error boundary."
    ), e0 = {
      then: function() {
        console.error(
          'Internal React error: A listener was unexpectedly attached to a "noop" thenable. This is a bug in React. Please file an issue.'
        );
      }
    }, ep = null, Qv = !1, yu = 0, mu = 1, qa = 2, aa = 4, jl = 8, y1 = 0, m1 = 1, p1 = 2, t0 = 3, cf = !1, v1 = !1, l0 = null, a0 = !1, hh = Ot(null), Zv = Ot(0), yh, g1 = /* @__PURE__ */ new Set(), b1 = /* @__PURE__ */ new Set(), n0 = /* @__PURE__ */ new Set(), S1 = /* @__PURE__ */ new Set(), of = 0, Be = null, xt = null, Al = null, Kv = !1, mh = !1, kr = !1, Jv = 0, tp = 0, Qc = null, LS = 0, VS = 25, G = null, pu = null, Zc = -1, lp = !1, kv = {
      readContext: Nt,
      use: jn,
      useCallback: Vt,
      useContext: Vt,
      useEffect: Vt,
      useImperativeHandle: Vt,
      useLayoutEffect: Vt,
      useInsertionEffect: Vt,
      useMemo: Vt,
      useReducer: Vt,
      useRef: Vt,
      useState: Vt,
      useDebugValue: Vt,
      useDeferredValue: Vt,
      useTransition: Vt,
      useSyncExternalStore: Vt,
      useId: Vt,
      useHostTransitionStatus: Vt,
      useFormState: Vt,
      useActionState: Vt,
      useOptimistic: Vt,
      useMemoCache: Vt,
      useCacheRefresh: Vt
    }, u0 = null, T1 = null, i0 = null, E1 = null, Li = null, Wu = null, $v = null;
    u0 = {
      readContext: function(e) {
        return Nt(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return G = "useCallback", $e(), Xa(t), kf(e, t);
      },
      useContext: function(e) {
        return G = "useContext", $e(), Nt(e);
      },
      useEffect: function(e, t) {
        return G = "useEffect", $e(), Xa(t), Ns(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", $e(), Xa(a), qs(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        G = "useInsertionEffect", $e(), Xa(t), Ja(4, qa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", $e(), Xa(t), ws(e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", $e(), Xa(t);
        var a = Y.H;
        Y.H = Li;
        try {
          return Bs(e, t);
        } finally {
          Y.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", $e();
        var i = Y.H;
        Y.H = Li;
        try {
          return rt(e, t, a);
        } finally {
          Y.H = i;
        }
      },
      useRef: function(e) {
        return G = "useRef", $e(), Jf(e);
      },
      useState: function(e) {
        G = "useState", $e();
        var t = Y.H;
        Y.H = Li;
        try {
          return Uu(e);
        } finally {
          Y.H = t;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", $e();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", $e(), Ys(e, t);
      },
      useTransition: function() {
        return G = "useTransition", $e(), Xn();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", $e(), _u(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", $e(), Qn();
      },
      useFormState: function(e, t) {
        return G = "useFormState", $e(), vo(), Ro(e, t);
      },
      useActionState: function(e, t) {
        return G = "useActionState", $e(), Ro(e, t);
      },
      useOptimistic: function(e) {
        return G = "useOptimistic", $e(), vn(e);
      },
      useHostTransitionStatus: sa,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", $e(), mc();
      }
    }, T1 = {
      readContext: function(e) {
        return Nt(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return G = "useCallback", ee(), kf(e, t);
      },
      useContext: function(e) {
        return G = "useContext", ee(), Nt(e);
      },
      useEffect: function(e, t) {
        return G = "useEffect", ee(), Ns(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", ee(), qs(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        G = "useInsertionEffect", ee(), Ja(4, qa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", ee(), ws(e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", ee();
        var a = Y.H;
        Y.H = Li;
        try {
          return Bs(e, t);
        } finally {
          Y.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", ee();
        var i = Y.H;
        Y.H = Li;
        try {
          return rt(e, t, a);
        } finally {
          Y.H = i;
        }
      },
      useRef: function(e) {
        return G = "useRef", ee(), Jf(e);
      },
      useState: function(e) {
        G = "useState", ee();
        var t = Y.H;
        Y.H = Li;
        try {
          return Uu(e);
        } finally {
          Y.H = t;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", ee(), Ys(e, t);
      },
      useTransition: function() {
        return G = "useTransition", ee(), Xn();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", ee(), _u(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", ee(), Qn();
      },
      useActionState: function(e, t) {
        return G = "useActionState", ee(), Ro(e, t);
      },
      useFormState: function(e, t) {
        return G = "useFormState", ee(), vo(), Ro(e, t);
      },
      useOptimistic: function(e) {
        return G = "useOptimistic", ee(), vn(e);
      },
      useHostTransitionStatus: sa,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), mc();
      }
    }, i0 = {
      readContext: function(e) {
        return Nt(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return G = "useCallback", ee(), hc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", ee(), Nt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", ee(), sl(2048, jl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", ee(), Vn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", ee(), sl(4, qa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", ee(), sl(4, aa, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", ee();
        var a = Y.H;
        Y.H = Wu;
        try {
          return gi(e, t);
        } finally {
          Y.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", ee();
        var i = Y.H;
        Y.H = Wu;
        try {
          return Za(e, t, a);
        } finally {
          Y.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", ee(), it().memoizedState;
      },
      useState: function() {
        G = "useState", ee();
        var e = Y.H;
        Y.H = Wu;
        try {
          return Za(dt);
        } finally {
          Y.H = e;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", ee(), $f(e, t);
      },
      useTransition: function() {
        return G = "useTransition", ee(), Ls();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", ee(), Xf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", ee(), it().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", ee(), vo(), Hs(e);
      },
      useActionState: function(e) {
        return G = "useActionState", ee(), Hs(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", ee(), Cu(e, t);
      },
      useHostTransitionStatus: sa,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), it().memoizedState;
      }
    }, E1 = {
      readContext: function(e) {
        return Nt(e);
      },
      use: jn,
      useCallback: function(e, t) {
        return G = "useCallback", ee(), hc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", ee(), Nt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", ee(), sl(2048, jl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", ee(), Vn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", ee(), sl(4, qa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", ee(), sl(4, aa, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", ee();
        var a = Y.H;
        Y.H = $v;
        try {
          return gi(e, t);
        } finally {
          Y.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", ee();
        var i = Y.H;
        Y.H = $v;
        try {
          return dc(e, t, a);
        } finally {
          Y.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", ee(), it().memoizedState;
      },
      useState: function() {
        G = "useState", ee();
        var e = Y.H;
        Y.H = $v;
        try {
          return dc(dt);
        } finally {
          Y.H = e;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", ee(), js(e, t);
      },
      useTransition: function() {
        return G = "useTransition", ee(), Vs();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", ee(), Xf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", ee(), it().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", ee(), vo(), Ao(e);
      },
      useActionState: function(e) {
        return G = "useActionState", ee(), Ao(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", ee(), xs(e, t);
      },
      useHostTransitionStatus: sa,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), it().memoizedState;
      }
    }, Li = {
      readContext: function(e) {
        return k(), Nt(e);
      },
      use: function(e) {
        return j(), jn(e);
      },
      useCallback: function(e, t) {
        return G = "useCallback", j(), $e(), kf(e, t);
      },
      useContext: function(e) {
        return G = "useContext", j(), $e(), Nt(e);
      },
      useEffect: function(e, t) {
        return G = "useEffect", j(), $e(), Ns(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", j(), $e(), qs(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        G = "useInsertionEffect", j(), $e(), Ja(4, qa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", j(), $e(), ws(e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", j(), $e();
        var a = Y.H;
        Y.H = Li;
        try {
          return Bs(e, t);
        } finally {
          Y.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", j(), $e();
        var i = Y.H;
        Y.H = Li;
        try {
          return rt(e, t, a);
        } finally {
          Y.H = i;
        }
      },
      useRef: function(e) {
        return G = "useRef", j(), $e(), Jf(e);
      },
      useState: function(e) {
        G = "useState", j(), $e();
        var t = Y.H;
        Y.H = Li;
        try {
          return Uu(e);
        } finally {
          Y.H = t;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", j(), $e();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", j(), $e(), Ys(e, t);
      },
      useTransition: function() {
        return G = "useTransition", j(), $e(), Xn();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", j(), $e(), _u(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", j(), $e(), Qn();
      },
      useFormState: function(e, t) {
        return G = "useFormState", j(), $e(), Ro(e, t);
      },
      useActionState: function(e, t) {
        return G = "useActionState", j(), $e(), Ro(e, t);
      },
      useOptimistic: function(e) {
        return G = "useOptimistic", j(), $e(), vn(e);
      },
      useMemoCache: function(e) {
        return j(), el(e);
      },
      useHostTransitionStatus: sa,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", $e(), mc();
      }
    }, Wu = {
      readContext: function(e) {
        return k(), Nt(e);
      },
      use: function(e) {
        return j(), jn(e);
      },
      useCallback: function(e, t) {
        return G = "useCallback", j(), ee(), hc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", j(), ee(), Nt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", j(), ee(), sl(2048, jl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", j(), ee(), Vn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", j(), ee(), sl(4, qa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", j(), ee(), sl(4, aa, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", j(), ee();
        var a = Y.H;
        Y.H = Wu;
        try {
          return gi(e, t);
        } finally {
          Y.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", j(), ee();
        var i = Y.H;
        Y.H = Wu;
        try {
          return Za(e, t, a);
        } finally {
          Y.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", j(), ee(), it().memoizedState;
      },
      useState: function() {
        G = "useState", j(), ee();
        var e = Y.H;
        Y.H = Wu;
        try {
          return Za(dt);
        } finally {
          Y.H = e;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", j(), ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", j(), ee(), $f(e, t);
      },
      useTransition: function() {
        return G = "useTransition", j(), ee(), Ls();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", j(), ee(), Xf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", j(), ee(), it().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", j(), ee(), Hs(e);
      },
      useActionState: function(e) {
        return G = "useActionState", j(), ee(), Hs(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", j(), ee(), Cu(e, t);
      },
      useMemoCache: function(e) {
        return j(), el(e);
      },
      useHostTransitionStatus: sa,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), it().memoizedState;
      }
    }, $v = {
      readContext: function(e) {
        return k(), Nt(e);
      },
      use: function(e) {
        return j(), jn(e);
      },
      useCallback: function(e, t) {
        return G = "useCallback", j(), ee(), hc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", j(), ee(), Nt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", j(), ee(), sl(2048, jl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", j(), ee(), Vn(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", j(), ee(), sl(4, qa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", j(), ee(), sl(4, aa, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", j(), ee();
        var a = Y.H;
        Y.H = Wu;
        try {
          return gi(e, t);
        } finally {
          Y.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", j(), ee();
        var i = Y.H;
        Y.H = Wu;
        try {
          return dc(e, t, a);
        } finally {
          Y.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", j(), ee(), it().memoizedState;
      },
      useState: function() {
        G = "useState", j(), ee();
        var e = Y.H;
        Y.H = Wu;
        try {
          return dc(dt);
        } finally {
          Y.H = e;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", j(), ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", j(), ee(), js(e, t);
      },
      useTransition: function() {
        return G = "useTransition", j(), ee(), Vs();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", j(), ee(), Xf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", j(), ee(), it().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", j(), ee(), Ao(e);
      },
      useActionState: function(e) {
        return G = "useActionState", j(), ee(), Ao(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", j(), ee(), xs(e, t);
      },
      useMemoCache: function(e) {
        return j(), el(e);
      },
      useHostTransitionStatus: sa,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), it().memoizedState;
      }
    };
    var R1 = {
      react_stack_bottom_frame: function(e, t, a) {
        var i = Sa;
        Sa = !0;
        try {
          return e(t, a);
        } finally {
          Sa = i;
        }
      }
    }, c0 = R1.react_stack_bottom_frame.bind(R1), A1 = {
      react_stack_bottom_frame: function(e) {
        var t = Sa;
        Sa = !0;
        try {
          return e.render();
        } finally {
          Sa = t;
        }
      }
    }, O1 = A1.react_stack_bottom_frame.bind(A1), D1 = {
      react_stack_bottom_frame: function(e, t) {
        try {
          t.componentDidMount();
        } catch (a) {
          Me(e, e.return, a);
        }
      }
    }, o0 = D1.react_stack_bottom_frame.bind(
      D1
    ), z1 = {
      react_stack_bottom_frame: function(e, t, a, i, o) {
        try {
          t.componentDidUpdate(a, i, o);
        } catch (f) {
          Me(e, e.return, f);
        }
      }
    }, M1 = z1.react_stack_bottom_frame.bind(
      z1
    ), _1 = {
      react_stack_bottom_frame: function(e, t) {
        var a = t.stack;
        e.componentDidCatch(t.value, {
          componentStack: a !== null ? a : ""
        });
      }
    }, XS = _1.react_stack_bottom_frame.bind(
      _1
    ), U1 = {
      react_stack_bottom_frame: function(e, t, a) {
        try {
          a.componentWillUnmount();
        } catch (i) {
          Me(e, t, i);
        }
      }
    }, C1 = U1.react_stack_bottom_frame.bind(
      U1
    ), x1 = {
      react_stack_bottom_frame: function(e) {
        e.resourceKind != null && console.error(
          "Expected only SimpleEffects when enableUseEffectCRUDOverload is disabled, got %s",
          e.resourceKind
        );
        var t = e.create;
        return e = e.inst, t = t(), e.destroy = t;
      }
    }, QS = x1.react_stack_bottom_frame.bind(x1), H1 = {
      react_stack_bottom_frame: function(e, t, a) {
        try {
          a();
        } catch (i) {
          Me(e, t, i);
        }
      }
    }, ZS = H1.react_stack_bottom_frame.bind(H1), N1 = {
      react_stack_bottom_frame: function(e) {
        var t = e._init;
        return t(e._payload);
      }
    }, ff = N1.react_stack_bottom_frame.bind(N1), ph = null, ap = 0, Fe = null, f0, w1 = f0 = !1, q1 = {}, B1 = {}, Y1 = {};
    st = function(e, t, a) {
      if (a !== null && typeof a == "object" && a._store && (!a._store.validated && a.key == null || a._store.validated === 2)) {
        if (typeof a._store != "object")
          throw Error(
            "React Component in warnForMissingKey should have a _store. This error is likely caused by a bug in React. Please file an issue."
          );
        a._store.validated = 1;
        var i = de(e), o = i || "null";
        if (!q1[o]) {
          q1[o] = !0, a = a._owner, e = e._debugOwner;
          var f = "";
          e && typeof e.tag == "number" && (o = de(e)) && (f = `

Check the render method of \`` + o + "`."), f || i && (f = `

Check the top-level render call using <` + i + ">.");
          var d = "";
          a != null && e !== a && (i = null, typeof a.tag == "number" ? i = de(a) : typeof a.name == "string" && (i = a.name), i && (d = " It was passed a child from " + i + ".")), he(t, function() {
            console.error(
              'Each child in a list should have a unique "key" prop.%s%s See https://react.dev/link/warning-keys for more information.',
              f,
              d
            );
          });
        }
      }
    };
    var vh = Pf(!0), j1 = Pf(!1), vu = Ot(null), Vi = null, gh = 1, np = 2, Gl = Ot(0), G1 = {}, L1 = /* @__PURE__ */ new Set(), V1 = /* @__PURE__ */ new Set(), X1 = /* @__PURE__ */ new Set(), Q1 = /* @__PURE__ */ new Set(), Z1 = /* @__PURE__ */ new Set(), K1 = /* @__PURE__ */ new Set(), J1 = /* @__PURE__ */ new Set(), k1 = /* @__PURE__ */ new Set(), $1 = /* @__PURE__ */ new Set(), W1 = /* @__PURE__ */ new Set();
    Object.freeze(G1);
    var r0 = {
      enqueueSetState: function(e, t, a) {
        e = e._reactInternals;
        var i = ya(e), o = Bn(i);
        o.payload = t, a != null && (Sy(a), o.callback = a), t = yn(e, o, i), t !== null && (Jt(t, e, i), hi(t, e, i)), _n(e, i);
      },
      enqueueReplaceState: function(e, t, a) {
        e = e._reactInternals;
        var i = ya(e), o = Bn(i);
        o.tag = m1, o.payload = t, a != null && (Sy(a), o.callback = a), t = yn(e, o, i), t !== null && (Jt(t, e, i), hi(t, e, i)), _n(e, i);
      },
      enqueueForceUpdate: function(e, t) {
        e = e._reactInternals;
        var a = ya(e), i = Bn(a);
        i.tag = p1, t != null && (Sy(t), i.callback = t), t = yn(e, i, a), t !== null && (Jt(t, e, a), hi(t, e, a)), fe !== null && typeof fe.markForceUpdateScheduled == "function" && fe.markForceUpdateScheduled(e, a);
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
      } else if (typeof Pt == "object" && typeof Pt.emit == "function") {
        Pt.emit("uncaughtException", e);
        return;
      }
      console.error(e);
    }, bh = null, d0 = null, F1 = Error(
      "This is not a real error. It's an implementation detail of React's selective hydration feature. If this leaks into userspace, it's a bug in React. Please file an issue."
    ), Jl = !1, I1 = {}, P1 = {}, eb = {}, tb = {}, Sh = !1, lb = {}, h0 = {}, y0 = {
      dehydrated: null,
      treeContext: null,
      retryLane: 0,
      hydrationErrors: null
    }, ab = !1, nb = null;
    nb = /* @__PURE__ */ new Set();
    var Kc = !1, yl = !1, m0 = !1, ub = typeof WeakSet == "function" ? WeakSet : Set, kl = null, Th = null, Eh = null, Ol = null, an = !1, Fu = null, up = 8192, KS = {
      getCacheForType: function(e) {
        var t = Nt(Yl), a = t.data.get(e);
        return a === void 0 && (a = e(), t.data.set(e, a)), a;
      },
      getOwner: function() {
        return Ha;
      }
    };
    if (typeof Symbol == "function" && Symbol.for) {
      var ip = Symbol.for;
      ip("selector.component"), ip("selector.has_pseudo_class"), ip("selector.role"), ip("selector.test_id"), ip("selector.text");
    }
    var JS = [], kS = typeof WeakMap == "function" ? WeakMap : Map, An = 0, Ba = 2, Iu = 4, Jc = 0, cp = 1, Rh = 2, p0 = 3, $r = 4, Wv = 6, ib = 5, Et = An, wt = null, at = null, nt = 0, nn = 0, op = 1, Wr = 2, fp = 3, cb = 4, v0 = 5, Ah = 6, rp = 7, g0 = 8, Fr = 9, _t = nn, On = null, rf = !1, Oh = !1, b0 = !1, Xi = 0, il = Jc, sf = 0, df = 0, S0 = 0, Dn = 0, Ir = 0, sp = null, Ya = null, Fv = !1, T0 = 0, ob = 300, Iv = 1 / 0, fb = 500, dp = null, hf = null, $S = 0, WS = 1, FS = 2, Pr = 0, rb = 1, sb = 2, db = 3, IS = 4, E0 = 5, na = 0, yf = null, Dh = null, mf = 0, R0 = 0, A0 = null, hb = null, PS = 50, hp = 0, O0 = null, D0 = !1, Pv = !1, eT = 50, es = 0, yp = null, zh = !1, eg = null, yb = !1, mb = /* @__PURE__ */ new Set(), tT = {}, tg = null, Mh = null, z0 = !1, M0 = !1, lg = !1, _0 = !1, ts = 0, U0 = {};
    (function() {
      for (var e = 0; e < Jg.length; e++) {
        var t = Jg[e], a = t.toLowerCase();
        t = t[0].toUpperCase() + t.slice(1), fn(a, "on" + t);
      }
      fn(P0, "onAnimationEnd"), fn(e1, "onAnimationIteration"), fn(t1, "onAnimationStart"), fn("dblclick", "onDoubleClick"), fn("focusin", "onFocus"), fn("focusout", "onBlur"), fn(xS, "onTransitionRun"), fn(HS, "onTransitionStart"), fn(NS, "onTransitionCancel"), fn(l1, "onTransitionEnd");
    })(), ne("onMouseEnter", ["mouseout", "mouseover"]), ne("onMouseLeave", ["mouseout", "mouseover"]), ne("onPointerEnter", ["pointerout", "pointerover"]), ne("onPointerLeave", ["pointerout", "pointerover"]), te(
      "onChange",
      "change click focusin focusout input keydown keyup selectionchange".split(
        " "
      )
    ), te(
      "onSelect",
      "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(
        " "
      )
    ), te("onBeforeInput", [
      "compositionend",
      "keypress",
      "textInput",
      "paste"
    ]), te(
      "onCompositionEnd",
      "compositionend focusout keydown keypress keyup mousedown".split(" ")
    ), te(
      "onCompositionStart",
      "compositionstart focusout keydown keypress keyup mousedown".split(" ")
    ), te(
      "onCompositionUpdate",
      "compositionupdate focusout keydown keypress keyup mousedown".split(" ")
    );
    var mp = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(
      " "
    ), C0 = new Set(
      "beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(mp)
    ), ag = "_reactListening" + Math.random().toString(36).slice(2), pb = !1, vb = !1, ng = !1, gb = !1, ug = !1, ig = !1, bb = !1, cg = {}, lT = /\r\n?/g, aT = /\u0000|\uFFFD/g, ls = "http://www.w3.org/1999/xlink", x0 = "http://www.w3.org/XML/1998/namespace", nT = "javascript:throw new Error('React form unexpectedly submitted.')", uT = "suppressHydrationWarning", og = "$", fg = "/$", kc = "$?", pp = "$!", iT = 1, cT = 2, oT = 4, H0 = "F!", Sb = "F", Tb = "complete", fT = "style", $c = 0, _h = 1, rg = 2, N0 = null, w0 = null, Eb = { dialog: !0, webview: !0 }, q0 = null, Rb = typeof setTimeout == "function" ? setTimeout : void 0, rT = typeof clearTimeout == "function" ? clearTimeout : void 0, as = -1, Ab = typeof Promise == "function" ? Promise : void 0, sT = typeof queueMicrotask == "function" ? queueMicrotask : typeof Ab < "u" ? function(e) {
      return Ab.resolve(null).then(e).catch(im);
    } : Rb, B0 = null, ns = 0, vp = 1, Ob = 2, Db = 3, gu = 4, bu = /* @__PURE__ */ new Map(), zb = /* @__PURE__ */ new Set(), Wc = Ue.d;
    Ue.d = {
      f: function() {
        var e = Wc.f(), t = Oc();
        return e || t;
      },
      r: function(e) {
        var t = Ml(e);
        t !== null && t.tag === 5 && t.type === "form" ? vy(t) : Wc.r(e);
      },
      D: function(e) {
        Wc.D(e), hv("dns-prefetch", e, null);
      },
      C: function(e, t) {
        Wc.C(e, t), hv("preconnect", e, t);
      },
      L: function(e, t, a) {
        Wc.L(e, t, a);
        var i = Uh;
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
              f = Mi(e);
              break;
            case "script":
              f = xc(e);
          }
          bu.has(f) || (e = ke(
            {
              rel: "preload",
              href: t === "image" && a && a.imageSrcSet ? void 0 : e,
              as: t
            },
            a
          ), bu.set(f, e), i.querySelector(o) !== null || t === "style" && i.querySelector(
            nu(f)
          ) || t === "script" && i.querySelector(Hc(f)) || (t = i.createElement("link"), $t(t, "link", e), z(t), i.head.appendChild(t)));
        }
      },
      m: function(e, t) {
        Wc.m(e, t);
        var a = Uh;
        if (a && e) {
          var i = t && typeof t.as == "string" ? t.as : "script", o = 'link[rel="modulepreload"][as="' + Aa(i) + '"][href="' + Aa(e) + '"]', f = o;
          switch (i) {
            case "audioworklet":
            case "paintworklet":
            case "serviceworker":
            case "sharedworker":
            case "worker":
            case "script":
              f = xc(e);
          }
          if (!bu.has(f) && (e = ke({ rel: "modulepreload", href: e }, t), bu.set(f, e), a.querySelector(o) === null)) {
            switch (i) {
              case "audioworklet":
              case "paintworklet":
              case "serviceworker":
              case "sharedworker":
              case "worker":
              case "script":
                if (a.querySelector(Hc(f)))
                  return;
            }
            i = a.createElement("link"), $t(i, "link", e), z(i), a.head.appendChild(i);
          }
        }
      },
      X: function(e, t) {
        Wc.X(e, t);
        var a = Uh;
        if (a && e) {
          var i = m(a).hoistableScripts, o = xc(e), f = i.get(o);
          f || (f = a.querySelector(
            Hc(o)
          ), f || (e = ke({ src: e, async: !0 }, t), (t = bu.get(o)) && ym(e, t), f = a.createElement("script"), z(f), $t(f, "link", e), a.head.appendChild(f)), f = {
            type: "script",
            instance: f,
            count: 1,
            state: null
          }, i.set(o, f));
        }
      },
      S: function(e, t, a) {
        Wc.S(e, t, a);
        var i = Uh;
        if (i && e) {
          var o = m(i).hoistableStyles, f = Mi(e);
          t = t || "default";
          var d = o.get(f);
          if (!d) {
            var h = { loading: ns, preload: null };
            if (d = i.querySelector(
              nu(f)
            ))
              h.loading = vp | gu;
            else {
              e = ke(
                {
                  rel: "stylesheet",
                  href: e,
                  "data-precedence": t
                },
                a
              ), (a = bu.get(f)) && hm(e, a);
              var v = d = i.createElement("link");
              z(v), $t(v, "link", e), v._p = new Promise(function(b, q) {
                v.onload = b, v.onerror = q;
              }), v.addEventListener("load", function() {
                h.loading |= vp;
              }), v.addEventListener("error", function() {
                h.loading |= Ob;
              }), h.loading |= gu, Hd(d, t, i);
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
        Wc.M(e, t);
        var a = Uh;
        if (a && e) {
          var i = m(a).hoistableScripts, o = xc(e), f = i.get(o);
          f || (f = a.querySelector(
            Hc(o)
          ), f || (e = ke({ src: e, async: !0, type: "module" }, t), (t = bu.get(o)) && ym(e, t), f = a.createElement("script"), z(f), $t(f, "link", e), a.head.appendChild(f)), f = {
            type: "script",
            instance: f,
            count: 1,
            state: null
          }, i.set(o, f));
        }
      }
    };
    var Uh = typeof document > "u" ? null : document, sg = null, gp = null, Y0 = null, dg = null, us = qg, bp = {
      $$typeof: Pa,
      Provider: null,
      Consumer: null,
      _currentValue: us,
      _currentValue2: us,
      _threadCount: 0
    }, Mb = "%c%s%c ", _b = "background: #e6e6e6;background: light-dark(rgba(0,0,0,0.1), rgba(255,255,255,0.25));color: #000000;color: light-dark(#000000, #ffffff);border-radius: 2px", Ub = "", hg = " ", dT = Function.prototype.bind, Cb = !1, xb = null, Hb = null, Nb = null, wb = null, qb = null, Bb = null, Yb = null, jb = null, Gb = null;
    xb = function(e, t, a, i) {
      t = H(e, t), t !== null && (a = F(t.memoizedState, a, 0, i), t.memoizedState = a, t.baseState = a, e.memoizedProps = ke({}, e.memoizedProps), a = ca(e, 2), a !== null && Jt(a, e, 2));
    }, Hb = function(e, t, a) {
      t = H(e, t), t !== null && (a = re(t.memoizedState, a, 0), t.memoizedState = a, t.baseState = a, e.memoizedProps = ke({}, e.memoizedProps), a = ca(e, 2), a !== null && Jt(a, e, 2));
    }, Nb = function(e, t, a, i) {
      t = H(e, t), t !== null && (a = Re(t.memoizedState, a, i), t.memoizedState = a, t.baseState = a, e.memoizedProps = ke({}, e.memoizedProps), a = ca(e, 2), a !== null && Jt(a, e, 2));
    }, wb = function(e, t, a) {
      e.pendingProps = F(e.memoizedProps, t, 0, a), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ca(e, 2), t !== null && Jt(t, e, 2);
    }, qb = function(e, t) {
      e.pendingProps = re(e.memoizedProps, t, 0), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ca(e, 2), t !== null && Jt(t, e, 2);
    }, Bb = function(e, t, a) {
      e.pendingProps = Re(
        e.memoizedProps,
        t,
        a
      ), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ca(e, 2), t !== null && Jt(t, e, 2);
    }, Yb = function(e) {
      var t = ca(e, 2);
      t !== null && Jt(t, e, 2);
    }, jb = function(e) {
      Ne = e;
    }, Gb = function(e) {
      Ae = e;
    };
    var yg = !0, mg = null, j0 = !1, pf = null, vf = null, gf = null, Sp = /* @__PURE__ */ new Map(), Tp = /* @__PURE__ */ new Map(), bf = [], hT = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(
      " "
    ), pg = null;
    if (Dr.prototype.render = jd.prototype.render = function(e) {
      var t = this._internalRoot;
      if (t === null) throw Error("Cannot update an unmounted root.");
      var a = arguments;
      typeof a[1] == "function" ? console.error(
        "does not support the second callback argument. To execute a side effect after rendering, declare it in a component body with useEffect()."
      ) : He(a[1]) ? console.error(
        "You passed a container to the second argument of root.render(...). You don't need to pass it again since you already passed it to create the root."
      ) : typeof a[1] < "u" && console.error(
        "You passed a second argument to root.render(...) but it only accepts one argument."
      ), a = e;
      var i = t.current, o = ya(i);
      Tt(i, o, a, t, null, null);
    }, Dr.prototype.unmount = jd.prototype.unmount = function() {
      var e = arguments;
      if (typeof e[0] == "function" && console.error(
        "does not support a callback argument. To execute a side effect after rendering, declare it in a component body with useEffect()."
      ), e = this._internalRoot, e !== null) {
        this._internalRoot = null;
        var t = e.containerInfo;
        (Et & (Ba | Iu)) !== An && console.error(
          "Attempted to synchronously unmount a root while React was already rendering. React cannot finish unmounting the root until the current render has completed, which may lead to a race condition."
        ), Tt(e.current, 2, null, e, null, null), Oc(), t[qi] = null;
      }
    }, Dr.prototype.unstable_scheduleHydration = function(e) {
      if (e) {
        var t = Ef();
        e = { blockedOn: null, target: e, priority: t };
        for (var a = 0; a < bf.length && t !== 0 && t < bf[a].priority; a++) ;
        bf.splice(a, 0, e), a === 0 && bv(e);
      }
    }, function() {
      var e = zr.version;
      if (e !== "19.1.1")
        throw Error(
          `Incompatible React versions: The "react" and "react-dom" packages must have the exact same version. Instead got:
  - react:      ` + (e + `
  - react-dom:  19.1.1
Learn more: https://react.dev/warnings/version-mismatch`)
        );
    }(), typeof Map == "function" && Map.prototype != null && typeof Map.prototype.forEach == "function" && typeof Set == "function" && Set.prototype != null && typeof Set.prototype.clear == "function" && typeof Set.prototype.forEach == "function" || console.error(
      "React depends on Map and Set built-in types. Make sure that you load a polyfill in older browsers. https://react.dev/link/react-polyfills"
    ), Ue.findDOMNode = function(e) {
      var t = e._reactInternals;
      if (t === void 0)
        throw typeof e.render == "function" ? Error("Unable to find node on an unmounted component.") : (e = Object.keys(e).join(","), Error(
          "Argument appears to not be a ReactComponent. Keys: " + e
        ));
      return e = At(t), e = e !== null ? be(e) : null, e = e === null ? null : e.stateNode, e;
    }, !function() {
      var e = {
        bundleType: 1,
        version: "19.1.1",
        rendererPackageName: "react-dom",
        currentDispatcherRef: Y,
        reconcilerVersion: "19.1.1"
      };
      return e.overrideHookState = xb, e.overrideHookStateDeletePath = Hb, e.overrideHookStateRenamePath = Nb, e.overrideProps = wb, e.overridePropsDeletePath = qb, e.overridePropsRenamePath = Bb, e.scheduleUpdate = Yb, e.setErrorHandler = jb, e.setSuspenseHandler = Gb, e.scheduleRefresh = Oe, e.scheduleRoot = ue, e.setRefreshHandler = ot, e.getCurrentFiber = xg, e.getLaneLabelMap = Hg, e.injectProfilingHooks = cl, De(e);
    }() && S && window.top === window.self && (-1 < navigator.userAgent.indexOf("Chrome") && navigator.userAgent.indexOf("Edge") === -1 || -1 < navigator.userAgent.indexOf("Firefox"))) {
      var Lb = window.location.protocol;
      /^(https?|file):$/.test(Lb) && console.info(
        "%cDownload the React DevTools for a better development experience: https://react.dev/link/react-devtools" + (Lb === "file:" ? `
You might need to use a local HTTP server (instead of file://): https://react.dev/link/react-devtools-faq` : ""),
        "font-weight:bold"
      );
    }
    Op.createRoot = function(e, t) {
      if (!He(e))
        throw Error("Target container is not a DOM element.");
      Ev(e);
      var a = !1, i = "", o = Ty, f = Fp, d = Ks, h = null;
      return t != null && (t.hydrate ? console.warn(
        "hydrate through createRoot is deprecated. Use ReactDOMClient.hydrateRoot(container, <App />) instead."
      ) : typeof t == "object" && t !== null && t.$$typeof === Ci && console.error(
        `You passed a JSX element to createRoot. You probably meant to call root.render instead. Example usage:

  let root = createRoot(domContainer);
  root.render(<App />);`
      ), t.unstable_strictMode === !0 && (a = !0), t.identifierPrefix !== void 0 && (i = t.identifierPrefix), t.onUncaughtError !== void 0 && (o = t.onUncaughtError), t.onCaughtError !== void 0 && (f = t.onCaughtError), t.onRecoverableError !== void 0 && (d = t.onRecoverableError), t.unstable_transitionCallbacks !== void 0 && (h = t.unstable_transitionCallbacks)), t = vm(
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
      ), e[qi] = t.current, Py(e), new jd(t);
    }, Op.hydrateRoot = function(e, t, a) {
      if (!He(e))
        throw Error("Target container is not a DOM element.");
      Ev(e), t === void 0 && console.error(
        "Must provide initial children as second argument to hydrateRoot. Example usage: hydrateRoot(domContainer, <App />)"
      );
      var i = !1, o = "", f = Ty, d = Fp, h = Ks, v = null, b = null;
      return a != null && (a.unstable_strictMode === !0 && (i = !0), a.identifierPrefix !== void 0 && (o = a.identifierPrefix), a.onUncaughtError !== void 0 && (f = a.onUncaughtError), a.onCaughtError !== void 0 && (d = a.onCaughtError), a.onRecoverableError !== void 0 && (h = a.onRecoverableError), a.unstable_transitionCallbacks !== void 0 && (v = a.unstable_transitionCallbacks), a.formState !== void 0 && (b = a.formState)), t = vm(
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
      ), t.context = gm(null), a = t.current, i = ya(a), i = Dl(i), o = Bn(i), o.callback = null, yn(a, o, i), a = i, t.current.lanes = a, Su(t, a), Wa(t), e[qi] = t.current, Py(e), new Dr(t);
    }, Op.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  }()), Op;
}
var lS;
function UT() {
  if (lS) return bg.exports;
  lS = 1;
  function H() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) {
      if (Pt.env.NODE_ENV !== "production")
        throw new Error("^_^");
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(H);
      } catch (F) {
        console.error(F);
      }
    }
  }
  return Pt.env.NODE_ENV === "production" ? (H(), bg.exports = MT()) : bg.exports = _T(), bg.exports;
}
var CT = UT();
let fS = tl.createContext(
  /** @type {any} */
  null
);
function xT() {
  let H = tl.useContext(fS);
  if (!H) throw new Error("RenderContext not found");
  return H;
}
function HT() {
  return xT().model;
}
function zn(H) {
  let F = HT(), Re = tl.useSyncExternalStore(
    (re) => (F.on(`change:${H}`, re), () => F.off(`change:${H}`, re)),
    () => F.get(H)
  ), _ = tl.useCallback(
    (re) => {
      F.set(
        H,
        // @ts-expect-error - TS cannot correctly narrow type
        typeof re == "function" ? re(F.get(H)) : re
      ), F.save_changes();
    },
    [F, H]
  );
  return [Re, _];
}
function NT(H) {
  return ({ el: F, model: Re, experimental: _ }) => {
    let re = CT.createRoot(F);
    return re.render(
      tl.createElement(
        tl.StrictMode,
        null,
        tl.createElement(
          fS.Provider,
          { value: { model: Re, experimental: _ } },
          tl.createElement(H)
        )
      )
    ), () => re.unmount();
  };
}
const wT = ({
  policies: H,
  selectedPolicies: F,
  searchTerm: Re,
  policyTypeFilter: _,
  tagFilter: re,
  uiConfig: Ae,
  useApiSearch: Ne,
  searchDebounceMs: st,
  searchCompleted: j,
  onSelectionChange: k,
  onFilterChange: ie,
  onApiSearch: K
}) => {
  const [D, ue] = tl.useState(Re), [Oe, ot] = tl.useState(!1), [He, Pe] = tl.useState([]), Ct = tl.useRef(null);
  tl.useEffect(() => {
    He.length === 0 && H.length > 0 && Pe(H);
  }, [H, He.length]);
  const Ke = tl.useMemo(() => {
    if (Ne)
      return H.slice(0, Ae.maxDisplayedPolicies);
    let le = H;
    if (D) {
      const R = D.toLowerCase();
      le = le.filter((X) => X.name.toLowerCase().includes(R));
    }
    return _.length > 0 && (le = le.filter((R) => _.includes(R.type))), re.length > 0 && (le = le.filter((R) => re.some((X) => R.tags.includes(X)))), le.slice(0, Ae.maxDisplayedPolicies);
  }, [H, D, _, re, Ae.maxDisplayedPolicies, Ne]), At = He.length > 0 ? He : H, be = tl.useMemo(() => {
    const le = new Set(At.map((R) => R.type));
    return Array.from(le).sort();
  }, [At]), pt = tl.useMemo(() => {
    const le = new Set(At.flatMap((R) => R.tags));
    return Array.from(le).sort();
  }, [At]), je = tl.useCallback(
    (le) => {
      const R = {
        searchTerm: le,
        policyTypeFilter: _,
        tagFilter: re
      };
      Ne && K ? (ot(!0), K(R)) : (ot(!1), ie(R));
    },
    [_, re, Ne, K, ie]
  ), St = (le) => {
    ue(le), Ct.current && clearTimeout(Ct.current), Ne && K ? Ct.current = setTimeout(() => {
      je(le);
    }, st) : ie({ searchTerm: le, policyTypeFilter: _, tagFilter: re });
  };
  tl.useEffect(() => () => {
    Ct.current && clearTimeout(Ct.current);
  }, []), tl.useEffect(() => {
    Oe && ot(!1);
  }, [H]), tl.useEffect(() => {
    j && Oe && ot(!1);
  }, [j, Oe]), tl.useEffect(() => {
    if (Oe) {
      const le = setTimeout(() => {
        ot(!1);
      }, 5e3);
      return () => clearTimeout(le);
    }
  }, [Oe]), tl.useEffect(() => {
    !Ne && Oe && ot(!1);
  }, [Ne, Oe]);
  const de = (le) => {
    ie({ searchTerm: D, policyTypeFilter: le, tagFilter: re });
  }, Ot = (le) => {
    ie({ searchTerm: D, policyTypeFilter: _, tagFilter: le });
  }, ve = (le) => {
    const R = F.includes(le) ? F.filter((X) => X !== le) : [...F, le];
    k(R);
  }, ze = () => {
    const le = Ke.map((R) => R.id);
    k([.../* @__PURE__ */ new Set([...F, ...le])]);
  }, Dt = () => {
    k([]);
  }, Ht = (le) => {
    try {
      return new Date(le).toLocaleDateString();
    } catch {
      return le;
    }
  };
  return /* @__PURE__ */ Rt.jsxs("div", { className: "policy-selector-widget", children: [
    /* @__PURE__ */ Rt.jsx("div", { className: "policy-selector-header", children: /* @__PURE__ */ Rt.jsx("h3", { className: "policy-selector-title", children: "Policy Selector" }) }),
    /* @__PURE__ */ Rt.jsxs("div", { className: "policy-selector-search", children: [
      /* @__PURE__ */ Rt.jsxs("div", { className: "search-input-container", children: [
        /* @__PURE__ */ Rt.jsx(
          "input",
          {
            type: "text",
            className: `search-input ${Oe ? "searching" : ""}`,
            placeholder: Ne ? "Search policies (API)..." : "Search policies by name...",
            value: D,
            onChange: (le) => St(le.target.value)
          }
        ),
        Oe && /* @__PURE__ */ Rt.jsx("div", { className: "search-loading-spinner" })
      ] }),
      /* @__PURE__ */ Rt.jsxs(
        "select",
        {
          className: "filter-dropdown",
          value: _.length > 0 ? _[0] : "",
          onChange: (le) => {
            const R = le.target.value, X = R ? [R] : [];
            Ne && K ? K({
              searchTerm: D,
              policyTypeFilter: X,
              tagFilter: re
            }) : de(X);
          },
          children: [
            /* @__PURE__ */ Rt.jsx("option", { value: "", children: "All Types" }),
            be.map((le) => /* @__PURE__ */ Rt.jsx("option", { value: le, children: le === "training_run" ? "Training Run" : "Policy" }, le))
          ]
        }
      ),
      /* @__PURE__ */ Rt.jsxs(
        "select",
        {
          className: "filter-dropdown",
          value: re.length > 0 ? re[0] : "",
          onChange: (le) => {
            const R = le.target.value, X = R ? [R] : [];
            Ne && K ? K({
              searchTerm: D,
              policyTypeFilter: _,
              tagFilter: X
            }) : Ot(X);
          },
          children: [
            /* @__PURE__ */ Rt.jsx("option", { value: "", children: "All Tags" }),
            pt.map((le) => /* @__PURE__ */ Rt.jsx("option", { value: le, children: le }, le))
          ]
        }
      )
    ] }),
    /* @__PURE__ */ Rt.jsxs("div", { className: "selection-controls", children: [
      /* @__PURE__ */ Rt.jsx("button", { className: "selection-button", onClick: ze, children: "Select All Filtered" }),
      /* @__PURE__ */ Rt.jsx("button", { className: "selection-button", onClick: Dt, children: "Clear Selection" }),
      /* @__PURE__ */ Rt.jsxs("div", { className: "selection-count", children: [
        F.length,
        " selected, ",
        Ke.length,
        " shown"
      ] })
    ] }),
    /* @__PURE__ */ Rt.jsx("div", { className: "policy-list", children: Ke.length === 0 ? /* @__PURE__ */ Rt.jsxs("div", { className: "policy-list-empty", children: [
      /* @__PURE__ */ Rt.jsx("div", { className: "policy-list-empty-icon", children: "🔍" }),
      /* @__PURE__ */ Rt.jsx("div", { children: H.length === 0 ? "No policies available" : "No policies match your filters" })
    ] }) : Ke.map((le) => /* @__PURE__ */ Rt.jsxs(
      "div",
      {
        className: `policy-item ${F.includes(le.id) ? "selected" : ""}`,
        onClick: () => ve(le.id),
        children: [
          /* @__PURE__ */ Rt.jsx(
            "input",
            {
              type: "checkbox",
              className: "policy-checkbox",
              checked: F.includes(le.id),
              onChange: () => ve(le.id),
              onClick: (R) => R.stopPropagation()
            }
          ),
          /* @__PURE__ */ Rt.jsxs("div", { className: "policy-info", children: [
            /* @__PURE__ */ Rt.jsx("div", { className: "policy-name", children: le.name }),
            /* @__PURE__ */ Rt.jsxs("div", { className: "policy-meta", children: [
              Ae.showType && /* @__PURE__ */ Rt.jsx("span", { className: `policy-type ${le.type}`, children: le.type === "training_run" ? "Training Run" : "Policy" }),
              Ae.showTags && le.tags.length > 0 && /* @__PURE__ */ Rt.jsx("div", { className: "policy-tags", children: le.tags.map((R) => /* @__PURE__ */ Rt.jsx("span", { className: "policy-tag", children: R }, R)) }),
              Ae.showCreatedAt && /* @__PURE__ */ Rt.jsx("span", { className: "policy-created-at", children: Ht(le.created_at) })
            ] })
          ] })
        ]
      },
      le.id
    )) })
  ] });
};
function qT() {
  const [H] = zn("policy_data"), [F, Re] = zn("selected_policies"), [_, re] = zn("search_term"), [Ae, Ne] = zn("policy_type_filter"), [st, j] = zn("tag_filter"), [k] = zn("use_api_search"), [ie] = zn("search_debounce_ms"), [K] = zn("api_search_completed"), [, D] = zn("selection_changed"), [, ue] = zn("filter_changed"), [, Oe] = zn("search_trigger"), [, ot] = zn("current_search_params"), [, He] = zn("load_all_policies_requested"), Pe = {
    showTags: !0,
    showType: !0,
    showCreatedAt: !0,
    maxDisplayedPolicies: 100
  }, Ct = (be) => {
    Re(be), D({
      selected_policies: be,
      action: "selection_updated",
      timestamp: Date.now()
    });
  }, Ke = (be) => {
    re(be.searchTerm || ""), Ne(be.policyTypeFilter || []), j(be.tagFilter || []), ue({
      ...be,
      timestamp: Date.now()
    });
  }, At = (be) => {
    console.log("🚀 React sending API search request to Python:", be);
    const pt = {
      ...be,
      timestamp: Date.now()
    };
    Ke(be), ot(pt), Oe((je) => (je || 0) + 1), console.log("🔢 Set search_trigger and current_search_params");
  };
  return tl.useEffect(() => {
    console.log("🚀 Loading all policies from client on mount"), He(!0);
  }, [He]), /* @__PURE__ */ Rt.jsx(
    wT,
    {
      policies: H || [],
      selectedPolicies: F || [],
      searchTerm: _ || "",
      policyTypeFilter: Ae || [],
      tagFilter: st || [],
      uiConfig: Pe,
      useApiSearch: k || !1,
      searchDebounceMs: ie || 300,
      searchCompleted: K,
      onSelectionChange: Ct,
      onFilterChange: Ke,
      onApiSearch: At
    }
  );
}
const BT = {
  render: NT(qT)
};
export {
  BT as default
};
