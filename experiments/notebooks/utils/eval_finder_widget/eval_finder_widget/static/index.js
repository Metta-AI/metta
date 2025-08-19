function mT(M) {
  return M && M.__esModule && Object.prototype.hasOwnProperty.call(M, "default") ? M.default : M;
}
var aS = { exports: {} }, ml = aS.exports = {}, Xi, Qi;
function V0() {
  throw new Error("setTimeout has not been defined");
}
function X0() {
  throw new Error("clearTimeout has not been defined");
}
(function() {
  try {
    typeof setTimeout == "function" ? Xi = setTimeout : Xi = V0;
  } catch {
    Xi = V0;
  }
  try {
    typeof clearTimeout == "function" ? Qi = clearTimeout : Qi = X0;
  } catch {
    Qi = X0;
  }
})();
function nS(M) {
  if (Xi === setTimeout)
    return setTimeout(M, 0);
  if ((Xi === V0 || !Xi) && setTimeout)
    return Xi = setTimeout, setTimeout(M, 0);
  try {
    return Xi(M, 0);
  } catch {
    try {
      return Xi.call(null, M, 0);
    } catch {
      return Xi.call(this, M, 0);
    }
  }
}
function pT(M) {
  if (Qi === clearTimeout)
    return clearTimeout(M);
  if ((Qi === X0 || !Qi) && clearTimeout)
    return Qi = clearTimeout, clearTimeout(M);
  try {
    return Qi(M);
  } catch {
    try {
      return Qi.call(null, M);
    } catch {
      return Qi.call(this, M);
    }
  }
}
var Fc = [], Ch = !1, is, Eg = -1;
function vT() {
  !Ch || !is || (Ch = !1, is.length ? Fc = is.concat(Fc) : Eg = -1, Fc.length && uS());
}
function uS() {
  if (!Ch) {
    var M = nS(vT);
    Ch = !0;
    for (var F = Fc.length; F; ) {
      for (is = Fc, Fc = []; ++Eg < F; )
        is && is[Eg].run();
      Eg = -1, F = Fc.length;
    }
    is = null, Ch = !1, pT(M);
  }
}
ml.nextTick = function(M) {
  var F = new Array(arguments.length - 1);
  if (arguments.length > 1)
    for (var re = 1; re < arguments.length; re++)
      F[re - 1] = arguments[re];
  Fc.push(new iS(M, F)), Fc.length === 1 && !Ch && nS(uS);
};
function iS(M, F) {
  this.fun = M, this.array = F;
}
iS.prototype.run = function() {
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
ml.listeners = function(M) {
  return [];
};
ml.binding = function(M) {
  throw new Error("process.binding is not supported");
};
ml.cwd = function() {
  return "/";
};
ml.chdir = function(M) {
  throw new Error("process.chdir is not supported");
};
ml.umask = function() {
  return 0;
};
var gT = aS.exports;
const It = /* @__PURE__ */ mT(gT);
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
  var M = Symbol.for("react.transitional.element"), F = Symbol.for("react.fragment");
  function re(_, ie, he) {
    var Oe = null;
    if (he !== void 0 && (Oe = "" + he), ie.key !== void 0 && (Oe = "" + ie.key), "key" in ie) {
      he = {};
      for (var Se in ie)
        Se !== "key" && (he[Se] = ie[Se]);
    } else he = ie;
    return ie = he.ref, {
      $$typeof: M,
      type: _,
      key: Oe,
      ref: ie !== void 0 ? ie : null,
      props: he
    };
  }
  return Ep.Fragment = F, Ep.jsx = re, Ep.jsxs = re, Ep;
}
var Ap = {}, gg = { exports: {} }, Pe = {}, Xb;
function ST() {
  if (Xb) return Pe;
  Xb = 1;
  var M = Symbol.for("react.transitional.element"), F = Symbol.for("react.portal"), re = Symbol.for("react.fragment"), _ = Symbol.for("react.strict_mode"), ie = Symbol.for("react.profiler"), he = Symbol.for("react.consumer"), Oe = Symbol.for("react.context"), Se = Symbol.for("react.forward_ref"), N = Symbol.for("react.suspense"), V = Symbol.for("react.memo"), le = Symbol.for("react.lazy"), k = Symbol.iterator;
  function U(g) {
    return g === null || typeof g != "object" ? null : (g = k && g[k] || g["@@iterator"], typeof g == "function" ? g : null);
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
  }, Ye = Object.assign, Mt = {};
  function $e(g, q, K) {
    this.props = g, this.context = q, this.refs = Mt, this.updater = K || ae;
  }
  $e.prototype.isReactComponent = {}, $e.prototype.setState = function(g, q) {
    if (typeof g != "object" && typeof g != "function" && g != null)
      throw Error(
        "takes an object of state variables to update or a function which returns an object of state variables."
      );
    this.updater.enqueueSetState(this, g, q, "setState");
  }, $e.prototype.forceUpdate = function(g) {
    this.updater.enqueueForceUpdate(this, g, "forceUpdate");
  };
  function tt() {
  }
  tt.prototype = $e.prototype;
  function Pt(g, q, K) {
    this.props = g, this.context = q, this.refs = Mt, this.updater = K || ae;
  }
  var Me = Pt.prototype = new tt();
  Me.constructor = Pt, Ye(Me, $e.prototype), Me.isPureReactComponent = !0;
  var lt = Array.isArray, De = { H: null, A: null, T: null, S: null, V: null }, bt = Object.prototype.hasOwnProperty;
  function Ge(g, q, K, I, ce, ze) {
    return K = ze.ref, {
      $$typeof: M,
      type: g,
      key: q,
      ref: K !== void 0 ? K : null,
      props: ze
    };
  }
  function Tt(g, q) {
    return Ge(
      g.type,
      q,
      void 0,
      void 0,
      void 0,
      g.props
    );
  }
  function de(g) {
    return typeof g == "object" && g !== null && g.$$typeof === M;
  }
  function Rt(g) {
    var q = { "=": "=0", ":": "=2" };
    return "$" + g.replace(/[=:]/g, function(K) {
      return q[K];
    });
  }
  var Te = /\/+/g;
  function Ce(g, q) {
    return typeof g == "object" && g !== null && g.key != null ? Rt("" + g.key) : q.toString(36);
  }
  function _t() {
  }
  function Gt(g) {
    switch (g.status) {
      case "fulfilled":
        return g.value;
      case "rejected":
        throw g.reason;
      default:
        switch (typeof g.status == "string" ? g.then(_t, _t) : (g.status = "pending", g.then(
          function(q) {
            g.status === "pending" && (g.status = "fulfilled", g.value = q);
          },
          function(q) {
            g.status === "pending" && (g.status = "rejected", g.reason = q);
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
  function pt(g, q, K, I, ce) {
    var ze = typeof g;
    (ze === "undefined" || ze === "boolean") && (g = null);
    var oe = !1;
    if (g === null) oe = !0;
    else
      switch (ze) {
        case "bigint":
        case "string":
        case "number":
          oe = !0;
          break;
        case "object":
          switch (g.$$typeof) {
            case M:
            case F:
              oe = !0;
              break;
            case le:
              return oe = g._init, pt(
                oe(g._payload),
                q,
                K,
                I,
                ce
              );
          }
      }
    if (oe)
      return ce = ce(g), oe = I === "" ? "." + Ce(g, 0) : I, lt(ce) ? (K = "", oe != null && (K = oe.replace(Te, "$&/") + "/"), pt(ce, q, K, "", function(wt) {
        return wt;
      })) : ce != null && (de(ce) && (ce = Tt(
        ce,
        K + (ce.key == null || g && g.key === ce.key ? "" : ("" + ce.key).replace(
          Te,
          "$&/"
        ) + "/") + oe
      )), q.push(ce)), 1;
    oe = 0;
    var il = I === "" ? "." : I + ":";
    if (lt(g))
      for (var Ne = 0; Ne < g.length; Ne++)
        I = g[Ne], ze = il + Ce(I, Ne), oe += pt(
          I,
          q,
          K,
          ze,
          ce
        );
    else if (Ne = U(g), typeof Ne == "function")
      for (g = Ne.call(g), Ne = 0; !(I = g.next()).done; )
        I = I.value, ze = il + Ce(I, Ne++), oe += pt(
          I,
          q,
          K,
          ze,
          ce
        );
    else if (ze === "object") {
      if (typeof g.then == "function")
        return pt(
          Gt(g),
          q,
          K,
          I,
          ce
        );
      throw q = String(g), Error(
        "Objects are not valid as a React child (found: " + (q === "[object Object]" ? "object with keys {" + Object.keys(g).join(", ") + "}" : q) + "). If you meant to render a collection of children, use an array instead."
      );
    }
    return oe;
  }
  function O(g, q, K) {
    if (g == null) return g;
    var I = [], ce = 0;
    return pt(g, I, "", "", function(ze) {
      return q.call(K, ze, ce++);
    }), I;
  }
  function W(g) {
    if (g._status === -1) {
      var q = g._result;
      q = q(), q.then(
        function(K) {
          (g._status === 0 || g._status === -1) && (g._status = 1, g._result = K);
        },
        function(K) {
          (g._status === 0 || g._status === -1) && (g._status = 2, g._result = K);
        }
      ), g._status === -1 && (g._status = 0, g._result = q);
    }
    if (g._status === 1) return g._result.default;
    throw g._result;
  }
  var P = typeof reportError == "function" ? reportError : function(g) {
    if (typeof window == "object" && typeof window.ErrorEvent == "function") {
      var q = new window.ErrorEvent("error", {
        bubbles: !0,
        cancelable: !0,
        message: typeof g == "object" && g !== null && typeof g.message == "string" ? String(g.message) : String(g),
        error: g
      });
      if (!window.dispatchEvent(q)) return;
    } else if (typeof It == "object" && typeof It.emit == "function") {
      It.emit("uncaughtException", g);
      return;
    }
    console.error(g);
  };
  function be() {
  }
  return Pe.Children = {
    map: O,
    forEach: function(g, q, K) {
      O(
        g,
        function() {
          q.apply(this, arguments);
        },
        K
      );
    },
    count: function(g) {
      var q = 0;
      return O(g, function() {
        q++;
      }), q;
    },
    toArray: function(g) {
      return O(g, function(q) {
        return q;
      }) || [];
    },
    only: function(g) {
      if (!de(g))
        throw Error(
          "React.Children.only expected to receive a single React element child."
        );
      return g;
    }
  }, Pe.Component = $e, Pe.Fragment = re, Pe.Profiler = ie, Pe.PureComponent = Pt, Pe.StrictMode = _, Pe.Suspense = N, Pe.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = De, Pe.__COMPILER_RUNTIME = {
    __proto__: null,
    c: function(g) {
      return De.H.useMemoCache(g);
    }
  }, Pe.cache = function(g) {
    return function() {
      return g.apply(null, arguments);
    };
  }, Pe.cloneElement = function(g, q, K) {
    if (g == null)
      throw Error(
        "The argument must be a React element, but you passed " + g + "."
      );
    var I = Ye({}, g.props), ce = g.key, ze = void 0;
    if (q != null)
      for (oe in q.ref !== void 0 && (ze = void 0), q.key !== void 0 && (ce = "" + q.key), q)
        !bt.call(q, oe) || oe === "key" || oe === "__self" || oe === "__source" || oe === "ref" && q.ref === void 0 || (I[oe] = q[oe]);
    var oe = arguments.length - 2;
    if (oe === 1) I.children = K;
    else if (1 < oe) {
      for (var il = Array(oe), Ne = 0; Ne < oe; Ne++)
        il[Ne] = arguments[Ne + 2];
      I.children = il;
    }
    return Ge(g.type, ce, void 0, void 0, ze, I);
  }, Pe.createContext = function(g) {
    return g = {
      $$typeof: Oe,
      _currentValue: g,
      _currentValue2: g,
      _threadCount: 0,
      Provider: null,
      Consumer: null
    }, g.Provider = g, g.Consumer = {
      $$typeof: he,
      _context: g
    }, g;
  }, Pe.createElement = function(g, q, K) {
    var I, ce = {}, ze = null;
    if (q != null)
      for (I in q.key !== void 0 && (ze = "" + q.key), q)
        bt.call(q, I) && I !== "key" && I !== "__self" && I !== "__source" && (ce[I] = q[I]);
    var oe = arguments.length - 2;
    if (oe === 1) ce.children = K;
    else if (1 < oe) {
      for (var il = Array(oe), Ne = 0; Ne < oe; Ne++)
        il[Ne] = arguments[Ne + 2];
      ce.children = il;
    }
    if (g && g.defaultProps)
      for (I in oe = g.defaultProps, oe)
        ce[I] === void 0 && (ce[I] = oe[I]);
    return Ge(g, ze, void 0, void 0, null, ce);
  }, Pe.createRef = function() {
    return { current: null };
  }, Pe.forwardRef = function(g) {
    return { $$typeof: Se, render: g };
  }, Pe.isValidElement = de, Pe.lazy = function(g) {
    return {
      $$typeof: le,
      _payload: { _status: -1, _result: g },
      _init: W
    };
  }, Pe.memo = function(g, q) {
    return {
      $$typeof: V,
      type: g,
      compare: q === void 0 ? null : q
    };
  }, Pe.startTransition = function(g) {
    var q = De.T, K = {};
    De.T = K;
    try {
      var I = g(), ce = De.S;
      ce !== null && ce(K, I), typeof I == "object" && I !== null && typeof I.then == "function" && I.then(be, P);
    } catch (ze) {
      P(ze);
    } finally {
      De.T = q;
    }
  }, Pe.unstable_useCacheRefresh = function() {
    return De.H.useCacheRefresh();
  }, Pe.use = function(g) {
    return De.H.use(g);
  }, Pe.useActionState = function(g, q, K) {
    return De.H.useActionState(g, q, K);
  }, Pe.useCallback = function(g, q) {
    return De.H.useCallback(g, q);
  }, Pe.useContext = function(g) {
    return De.H.useContext(g);
  }, Pe.useDebugValue = function() {
  }, Pe.useDeferredValue = function(g, q) {
    return De.H.useDeferredValue(g, q);
  }, Pe.useEffect = function(g, q, K) {
    var I = De.H;
    if (typeof K == "function")
      throw Error(
        "useEffect CRUD overload is not enabled in this build of React."
      );
    return I.useEffect(g, q);
  }, Pe.useId = function() {
    return De.H.useId();
  }, Pe.useImperativeHandle = function(g, q, K) {
    return De.H.useImperativeHandle(g, q, K);
  }, Pe.useInsertionEffect = function(g, q) {
    return De.H.useInsertionEffect(g, q);
  }, Pe.useLayoutEffect = function(g, q) {
    return De.H.useLayoutEffect(g, q);
  }, Pe.useMemo = function(g, q) {
    return De.H.useMemo(g, q);
  }, Pe.useOptimistic = function(g, q) {
    return De.H.useOptimistic(g, q);
  }, Pe.useReducer = function(g, q, K) {
    return De.H.useReducer(g, q, K);
  }, Pe.useRef = function(g) {
    return De.H.useRef(g);
  }, Pe.useState = function(g) {
    return De.H.useState(g);
  }, Pe.useSyncExternalStore = function(g, q, K) {
    return De.H.useSyncExternalStore(
      g,
      q,
      K
    );
  }, Pe.useTransition = function() {
    return De.H.useTransition();
  }, Pe.version = "19.1.1", Pe;
}
var Dp = { exports: {} };
Dp.exports;
var Qb;
function TT() {
  return Qb || (Qb = 1, function(M, F) {
    It.env.NODE_ENV !== "production" && function() {
      function re(m, D) {
        Object.defineProperty(he.prototype, m, {
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
        return m === null || typeof m != "object" ? null : (m = zn && m[zn] || m["@@iterator"], typeof m == "function" ? m : null);
      }
      function ie(m, D) {
        m = (m = m.constructor) && (m.displayName || m.name) || "ReactClass";
        var te = m + "." + D;
        Zi[te] || (console.error(
          "Can't call %s on a component that is not yet mounted. This is a no-op, but it might indicate a bug in your application. Instead, assign to `this.state` directly or define a `state = {};` class property with the desired state in the %s component.",
          D,
          m
        ), Zi[te] = !0);
      }
      function he(m, D, te) {
        this.props = m, this.context = D, this.refs = Sf, this.updater = te || Mn;
      }
      function Oe() {
      }
      function Se(m, D, te) {
        this.props = m, this.context = D, this.refs = Sf, this.updater = te || Mn;
      }
      function N(m) {
        return "" + m;
      }
      function V(m) {
        try {
          N(m);
          var D = !1;
        } catch {
          D = !0;
        }
        if (D) {
          D = console;
          var te = D.error, ue = typeof Symbol == "function" && Symbol.toStringTag && m[Symbol.toStringTag] || m.constructor.name || "Object";
          return te.call(
            D,
            "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
            ue
          ), N(m);
        }
      }
      function le(m) {
        if (m == null) return null;
        if (typeof m == "function")
          return m.$$typeof === cs ? null : m.displayName || m.name || null;
        if (typeof m == "string") return m;
        switch (m) {
          case g:
            return "Fragment";
          case K:
            return "Profiler";
          case q:
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
            case ze:
              var D = m.render;
              return m = m.displayName, m || (m = D.displayName || D.name || "", m = m !== "" ? "ForwardRef(" + m + ")" : "ForwardRef"), m;
            case Ne:
              return D = m.displayName || null, D !== null ? D : le(m.type) || "Memo";
            case wt:
              D = m._payload, m = m._init;
              try {
                return le(m(D));
              } catch {
              }
          }
        return null;
      }
      function k(m) {
        if (m === g) return "<>";
        if (typeof m == "object" && m !== null && m.$$typeof === wt)
          return "<...>";
        try {
          var D = le(m);
          return D ? "<" + D + ">" : "<...>";
        } catch {
          return "<...>";
        }
      }
      function U() {
        var m = Je.A;
        return m === null ? null : m.getOwner();
      }
      function ae() {
        return Error("react-stack-top-frame");
      }
      function Ye(m) {
        if (Un.call(m, "key")) {
          var D = Object.getOwnPropertyDescriptor(m, "key").get;
          if (D && D.isReactWarning) return !1;
        }
        return m.key !== void 0;
      }
      function Mt(m, D) {
        function te() {
          bu || (bu = !0, console.error(
            "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
            D
          ));
        }
        te.isReactWarning = !0, Object.defineProperty(m, "key", {
          get: te,
          configurable: !0
        });
      }
      function $e() {
        var m = le(this.type);
        return Tf[m] || (Tf[m] = !0, console.error(
          "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
        )), m = this.props.ref, m !== void 0 ? m : null;
      }
      function tt(m, D, te, ue, ve, we, Le, ct) {
        return te = we.ref, m = {
          $$typeof: P,
          type: m,
          key: D,
          props: we,
          _owner: ve
        }, (te !== void 0 ? te : null) !== null ? Object.defineProperty(m, "ref", {
          enumerable: !1,
          get: $e
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
          value: Le
        }), Object.defineProperty(m, "_debugTask", {
          configurable: !1,
          enumerable: !1,
          writable: !0,
          value: ct
        }), Object.freeze && (Object.freeze(m.props), Object.freeze(m)), m;
      }
      function Pt(m, D) {
        return D = tt(
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
      function Me(m) {
        return typeof m == "object" && m !== null && m.$$typeof === P;
      }
      function lt(m) {
        var D = { "=": "=0", ":": "=2" };
        return "$" + m.replace(/[=:]/g, function(te) {
          return D[te];
        });
      }
      function De(m, D) {
        return typeof m == "object" && m !== null && m.key != null ? (V(m.key), lt("" + m.key)) : D.toString(36);
      }
      function bt() {
      }
      function Ge(m) {
        switch (m.status) {
          case "fulfilled":
            return m.value;
          case "rejected":
            throw m.reason;
          default:
            switch (typeof m.status == "string" ? m.then(bt, bt) : (m.status = "pending", m.then(
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
      function Tt(m, D, te, ue, ve) {
        var we = typeof m;
        (we === "undefined" || we === "boolean") && (m = null);
        var Le = !1;
        if (m === null) Le = !0;
        else
          switch (we) {
            case "bigint":
            case "string":
            case "number":
              Le = !0;
              break;
            case "object":
              switch (m.$$typeof) {
                case P:
                case be:
                  Le = !0;
                  break;
                case wt:
                  return Le = m._init, Tt(
                    Le(m._payload),
                    D,
                    te,
                    ue,
                    ve
                  );
              }
          }
        if (Le) {
          Le = m, ve = ve(Le);
          var ct = ue === "" ? "." + De(Le, 0) : ue;
          return Iu(ve) ? (te = "", ct != null && (te = ct.replace(Dl, "$&/") + "/"), Tt(ve, D, te, "", function(ll) {
            return ll;
          })) : ve != null && (Me(ve) && (ve.key != null && (Le && Le.key === ve.key || V(ve.key)), te = Pt(
            ve,
            te + (ve.key == null || Le && Le.key === ve.key ? "" : ("" + ve.key).replace(
              Dl,
              "$&/"
            ) + "/") + ct
          ), ue !== "" && Le != null && Me(Le) && Le.key == null && Le._store && !Le._store.validated && (te._store.validated = 2), ve = te), D.push(ve)), 1;
        }
        if (Le = 0, ct = ue === "" ? "." : ue + ":", Iu(m))
          for (var je = 0; je < m.length; je++)
            ue = m[je], we = ct + De(ue, je), Le += Tt(
              ue,
              D,
              te,
              we,
              ve
            );
        else if (je = _(m), typeof je == "function")
          for (je === m.entries && (Ya || console.warn(
            "Using Maps as children is not supported. Use an array of keyed ReactElements instead."
          ), Ya = !0), m = je.call(m), je = 0; !(ue = m.next()).done; )
            ue = ue.value, we = ct + De(ue, je++), Le += Tt(
              ue,
              D,
              te,
              we,
              ve
            );
        else if (we === "object") {
          if (typeof m.then == "function")
            return Tt(
              Ge(m),
              D,
              te,
              ue,
              ve
            );
          throw D = String(m), Error(
            "Objects are not valid as a React child (found: " + (D === "[object Object]" ? "object with keys {" + Object.keys(m).join(", ") + "}" : D) + "). If you meant to render a collection of children, use an array instead."
          );
        }
        return Le;
      }
      function de(m, D, te) {
        if (m == null) return m;
        var ue = [], ve = 0;
        return Tt(m, ue, "", "", function(we) {
          return D.call(te, we, ve++);
        }), ue;
      }
      function Rt(m) {
        if (m._status === -1) {
          var D = m._result;
          D = D(), D.then(
            function(te) {
              (m._status === 0 || m._status === -1) && (m._status = 1, m._result = te);
            },
            function(te) {
              (m._status === 0 || m._status === -1) && (m._status = 2, m._result = te);
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
      function Te() {
        var m = Je.H;
        return m === null && console.error(
          `Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:
1. You might have mismatching versions of React and the renderer (such as React DOM)
2. You might be breaking the Rules of Hooks
3. You might have more than one copy of React in the same app
See https://react.dev/link/invalid-hook-call for tips about how to debug and fix this problem.`
        ), m;
      }
      function Ce() {
      }
      function _t(m) {
        if (lo === null)
          try {
            var D = ("require" + Math.random()).slice(0, 7);
            lo = (M && M[D]).call(
              M,
              "timers"
            ).setImmediate;
          } catch {
            lo = function(ue) {
              Ef === !1 && (Ef = !0, typeof MessageChannel > "u" && console.error(
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
      function O(m, D, te) {
        var ue = Je.actQueue;
        if (ue !== null)
          if (ue.length !== 0)
            try {
              W(ue), _t(function() {
                return O(m, D, te);
              });
              return;
            } catch (ve) {
              Je.thrownErrors.push(ve);
            }
          else Je.actQueue = null;
        0 < Je.thrownErrors.length ? (ue = Gt(Je.thrownErrors), Je.thrownErrors.length = 0, te(ue)) : D(m);
      }
      function W(m) {
        if (!zl) {
          zl = !0;
          var D = 0;
          try {
            for (; D < m.length; D++) {
              var te = m[D];
              do {
                Je.didUsePromise = !1;
                var ue = te(!1);
                if (ue !== null) {
                  if (Je.didUsePromise) {
                    m[D] = te, m.splice(0, D);
                    return;
                  }
                  te = ue;
                } else break;
              } while (!0);
            }
            m.length = 0;
          } catch (ve) {
            m.splice(0, D + 1), Je.thrownErrors.push(ve);
          } finally {
            zl = !1;
          }
        }
      }
      typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
      var P = Symbol.for("react.transitional.element"), be = Symbol.for("react.portal"), g = Symbol.for("react.fragment"), q = Symbol.for("react.strict_mode"), K = Symbol.for("react.profiler"), I = Symbol.for("react.consumer"), ce = Symbol.for("react.context"), ze = Symbol.for("react.forward_ref"), oe = Symbol.for("react.suspense"), il = Symbol.for("react.suspense_list"), Ne = Symbol.for("react.memo"), wt = Symbol.for("react.lazy"), na = Symbol.for("react.activity"), zn = Symbol.iterator, Zi = {}, Mn = {
        isMounted: function() {
          return !1;
        },
        enqueueForceUpdate: function(m) {
          ie(m, "forceUpdate");
        },
        enqueueReplaceState: function(m) {
          ie(m, "replaceState");
        },
        enqueueSetState: function(m) {
          ie(m, "setState");
        }
      }, Pc = Object.assign, Sf = {};
      Object.freeze(Sf), he.prototype.isReactComponent = {}, he.prototype.setState = function(m, D) {
        if (typeof m != "object" && typeof m != "function" && m != null)
          throw Error(
            "takes an object of state variables to update or a function which returns an object of state variables."
          );
        this.updater.enqueueSetState(this, m, D, "setState");
      }, he.prototype.forceUpdate = function(m) {
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
        tl.hasOwnProperty(pl) && re(pl, tl[pl]);
      Oe.prototype = he.prototype, tl = Se.prototype = new Oe(), tl.constructor = Se, Pc(tl, he.prototype), tl.isPureReactComponent = !0;
      var Iu = Array.isArray, cs = Symbol.for("react.client.reference"), Je = {
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
      }, Un = Object.prototype.hasOwnProperty, eo = console.createTask ? console.createTask : function() {
        return null;
      };
      tl = {
        react_stack_bottom_frame: function(m) {
          return m();
        }
      };
      var bu, os, Tf = {}, Pu = tl.react_stack_bottom_frame.bind(
        tl,
        ae
      )(), Ol = eo(k(ae)), Ya = !1, Dl = /\/+/g, to = typeof reportError == "function" ? reportError : function(m) {
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
      }, Ef = !1, lo = null, nn = 0, ua = !1, zl = !1, un = typeof queueMicrotask == "function" ? function(m) {
        queueMicrotask(function() {
          return queueMicrotask(m);
        });
      } : _t;
      tl = Object.freeze({
        __proto__: null,
        c: function(m) {
          return Te().useMemoCache(m);
        }
      }), F.Children = {
        map: de,
        forEach: function(m, D, te) {
          de(
            m,
            function() {
              D.apply(this, arguments);
            },
            te
          );
        },
        count: function(m) {
          var D = 0;
          return de(m, function() {
            D++;
          }), D;
        },
        toArray: function(m) {
          return de(m, function(D) {
            return D;
          }) || [];
        },
        only: function(m) {
          if (!Me(m))
            throw Error(
              "React.Children.only expected to receive a single React element child."
            );
          return m;
        }
      }, F.Component = he, F.Fragment = g, F.Profiler = K, F.PureComponent = Se, F.StrictMode = q, F.Suspense = oe, F.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = Je, F.__COMPILER_RUNTIME = tl, F.act = function(m) {
        var D = Je.actQueue, te = nn;
        nn++;
        var ue = Je.actQueue = D !== null ? D : [], ve = !1;
        try {
          var we = m();
        } catch (je) {
          Je.thrownErrors.push(je);
        }
        if (0 < Je.thrownErrors.length)
          throw pt(D, te), m = Gt(Je.thrownErrors), Je.thrownErrors.length = 0, m;
        if (we !== null && typeof we == "object" && typeof we.then == "function") {
          var Le = we;
          return un(function() {
            ve || ua || (ua = !0, console.error(
              "You called act(async () => ...) without await. This could lead to unexpected testing behaviour, interleaving multiple act calls and mixing their scopes. You should - await act(async () => ...);"
            ));
          }), {
            then: function(je, ll) {
              ve = !0, Le.then(
                function(cn) {
                  if (pt(D, te), te === 0) {
                    try {
                      W(ue), _t(function() {
                        return O(
                          cn,
                          je,
                          ll
                        );
                      });
                    } catch (Hh) {
                      Je.thrownErrors.push(Hh);
                    }
                    if (0 < Je.thrownErrors.length) {
                      var fs = Gt(
                        Je.thrownErrors
                      );
                      Je.thrownErrors.length = 0, ll(fs);
                    }
                  } else je(cn);
                },
                function(cn) {
                  pt(D, te), 0 < Je.thrownErrors.length && (cn = Gt(
                    Je.thrownErrors
                  ), Je.thrownErrors.length = 0), ll(cn);
                }
              );
            }
          };
        }
        var ct = we;
        if (pt(D, te), te === 0 && (W(ue), ue.length !== 0 && un(function() {
          ve || ua || (ua = !0, console.error(
            "A component suspended inside an `act` scope, but the `act` call was not awaited. When testing React components that depend on asynchronous data, you must await the result:\n\nawait act(() => ...)"
          ));
        }), Je.actQueue = null), 0 < Je.thrownErrors.length)
          throw m = Gt(Je.thrownErrors), Je.thrownErrors.length = 0, m;
        return {
          then: function(je, ll) {
            ve = !0, te === 0 ? (Je.actQueue = ue, _t(function() {
              return O(
                ct,
                je,
                ll
              );
            })) : je(ct);
          }
        };
      }, F.cache = function(m) {
        return function() {
          return m.apply(null, arguments);
        };
      }, F.captureOwnerStack = function() {
        var m = Je.getCurrentStack;
        return m === null ? null : m();
      }, F.cloneElement = function(m, D, te) {
        if (m == null)
          throw Error(
            "The argument must be a React element, but you passed " + m + "."
          );
        var ue = Pc({}, m.props), ve = m.key, we = m._owner;
        if (D != null) {
          var Le;
          e: {
            if (Un.call(D, "ref") && (Le = Object.getOwnPropertyDescriptor(
              D,
              "ref"
            ).get) && Le.isReactWarning) {
              Le = !1;
              break e;
            }
            Le = D.ref !== void 0;
          }
          Le && (we = U()), Ye(D) && (V(D.key), ve = "" + D.key);
          for (ct in D)
            !Un.call(D, ct) || ct === "key" || ct === "__self" || ct === "__source" || ct === "ref" && D.ref === void 0 || (ue[ct] = D[ct]);
        }
        var ct = arguments.length - 2;
        if (ct === 1) ue.children = te;
        else if (1 < ct) {
          Le = Array(ct);
          for (var je = 0; je < ct; je++)
            Le[je] = arguments[je + 2];
          ue.children = Le;
        }
        for (ue = tt(
          m.type,
          ve,
          void 0,
          void 0,
          we,
          ue,
          m._debugStack,
          m._debugTask
        ), ve = 2; ve < arguments.length; ve++)
          we = arguments[ve], Me(we) && we._store && (we._store.validated = 1);
        return ue;
      }, F.createContext = function(m) {
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
      }, F.createElement = function(m, D, te) {
        for (var ue = 2; ue < arguments.length; ue++) {
          var ve = arguments[ue];
          Me(ve) && ve._store && (ve._store.validated = 1);
        }
        if (ue = {}, ve = null, D != null)
          for (je in os || !("__self" in D) || "key" in D || (os = !0, console.warn(
            "Your app (or one of its dependencies) is using an outdated JSX transform. Update to the modern JSX transform for faster performance: https://react.dev/link/new-jsx-transform"
          )), Ye(D) && (V(D.key), ve = "" + D.key), D)
            Un.call(D, je) && je !== "key" && je !== "__self" && je !== "__source" && (ue[je] = D[je]);
        var we = arguments.length - 2;
        if (we === 1) ue.children = te;
        else if (1 < we) {
          for (var Le = Array(we), ct = 0; ct < we; ct++)
            Le[ct] = arguments[ct + 2];
          Object.freeze && Object.freeze(Le), ue.children = Le;
        }
        if (m && m.defaultProps)
          for (je in we = m.defaultProps, we)
            ue[je] === void 0 && (ue[je] = we[je]);
        ve && Mt(
          ue,
          typeof m == "function" ? m.displayName || m.name || "Unknown" : m
        );
        var je = 1e4 > Je.recentlyCreatedOwnerStacks++;
        return tt(
          m,
          ve,
          void 0,
          void 0,
          U(),
          ue,
          je ? Error("react-stack-top-frame") : Pu,
          je ? eo(k(m)) : Ol
        );
      }, F.createRef = function() {
        var m = { current: null };
        return Object.seal(m), m;
      }, F.forwardRef = function(m) {
        m != null && m.$$typeof === Ne ? console.error(
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
        var D = { $$typeof: ze, render: m }, te;
        return Object.defineProperty(D, "displayName", {
          enumerable: !1,
          configurable: !0,
          get: function() {
            return te;
          },
          set: function(ue) {
            te = ue, m.name || m.displayName || (Object.defineProperty(m, "name", { value: ue }), m.displayName = ue);
          }
        }), D;
      }, F.isValidElement = Me, F.lazy = function(m) {
        return {
          $$typeof: wt,
          _payload: { _status: -1, _result: m },
          _init: Rt
        };
      }, F.memo = function(m, D) {
        m == null && console.error(
          "memo: The first argument must be a component. Instead received: %s",
          m === null ? "null" : typeof m
        ), D = {
          $$typeof: Ne,
          type: m,
          compare: D === void 0 ? null : D
        };
        var te;
        return Object.defineProperty(D, "displayName", {
          enumerable: !1,
          configurable: !0,
          get: function() {
            return te;
          },
          set: function(ue) {
            te = ue, m.name || m.displayName || (Object.defineProperty(m, "name", { value: ue }), m.displayName = ue);
          }
        }), D;
      }, F.startTransition = function(m) {
        var D = Je.T, te = {};
        Je.T = te, te._updatedFibers = /* @__PURE__ */ new Set();
        try {
          var ue = m(), ve = Je.S;
          ve !== null && ve(te, ue), typeof ue == "object" && ue !== null && typeof ue.then == "function" && ue.then(Ce, to);
        } catch (we) {
          to(we);
        } finally {
          D === null && te._updatedFibers && (m = te._updatedFibers.size, te._updatedFibers.clear(), 10 < m && console.warn(
            "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
          )), Je.T = D;
        }
      }, F.unstable_useCacheRefresh = function() {
        return Te().useCacheRefresh();
      }, F.use = function(m) {
        return Te().use(m);
      }, F.useActionState = function(m, D, te) {
        return Te().useActionState(
          m,
          D,
          te
        );
      }, F.useCallback = function(m, D) {
        return Te().useCallback(m, D);
      }, F.useContext = function(m) {
        var D = Te();
        return m.$$typeof === I && console.error(
          "Calling useContext(Context.Consumer) is not supported and will cause bugs. Did you mean to call useContext(Context) instead?"
        ), D.useContext(m);
      }, F.useDebugValue = function(m, D) {
        return Te().useDebugValue(m, D);
      }, F.useDeferredValue = function(m, D) {
        return Te().useDeferredValue(m, D);
      }, F.useEffect = function(m, D, te) {
        m == null && console.warn(
          "React Hook useEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        );
        var ue = Te();
        if (typeof te == "function")
          throw Error(
            "useEffect CRUD overload is not enabled in this build of React."
          );
        return ue.useEffect(m, D);
      }, F.useId = function() {
        return Te().useId();
      }, F.useImperativeHandle = function(m, D, te) {
        return Te().useImperativeHandle(m, D, te);
      }, F.useInsertionEffect = function(m, D) {
        return m == null && console.warn(
          "React Hook useInsertionEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        ), Te().useInsertionEffect(m, D);
      }, F.useLayoutEffect = function(m, D) {
        return m == null && console.warn(
          "React Hook useLayoutEffect requires an effect callback. Did you forget to pass a callback to the hook?"
        ), Te().useLayoutEffect(m, D);
      }, F.useMemo = function(m, D) {
        return Te().useMemo(m, D);
      }, F.useOptimistic = function(m, D) {
        return Te().useOptimistic(m, D);
      }, F.useReducer = function(m, D, te) {
        return Te().useReducer(m, D, te);
      }, F.useRef = function(m) {
        return Te().useRef(m);
      }, F.useState = function(m) {
        return Te().useState(m);
      }, F.useSyncExternalStore = function(m, D, te) {
        return Te().useSyncExternalStore(
          m,
          D,
          te
        );
      }, F.useTransition = function() {
        return Te().useTransition();
      }, F.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    }();
  }(Dp, Dp.exports)), Dp.exports;
}
var Zb;
function xh() {
  return Zb || (Zb = 1, It.env.NODE_ENV === "production" ? gg.exports = ST() : gg.exports = TT()), gg.exports;
}
var Kb;
function ET() {
  return Kb || (Kb = 1, It.env.NODE_ENV !== "production" && function() {
    function M(g) {
      if (g == null) return null;
      if (typeof g == "function")
        return g.$$typeof === Rt ? null : g.displayName || g.name || null;
      if (typeof g == "string") return g;
      switch (g) {
        case Mt:
          return "Fragment";
        case tt:
          return "Profiler";
        case $e:
          return "StrictMode";
        case De:
          return "Suspense";
        case bt:
          return "SuspenseList";
        case de:
          return "Activity";
      }
      if (typeof g == "object")
        switch (typeof g.tag == "number" && console.error(
          "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
        ), g.$$typeof) {
          case Ye:
            return "Portal";
          case Me:
            return (g.displayName || "Context") + ".Provider";
          case Pt:
            return (g._context.displayName || "Context") + ".Consumer";
          case lt:
            var q = g.render;
            return g = g.displayName, g || (g = q.displayName || q.name || "", g = g !== "" ? "ForwardRef(" + g + ")" : "ForwardRef"), g;
          case Ge:
            return q = g.displayName || null, q !== null ? q : M(g.type) || "Memo";
          case Tt:
            q = g._payload, g = g._init;
            try {
              return M(g(q));
            } catch {
            }
        }
      return null;
    }
    function F(g) {
      return "" + g;
    }
    function re(g) {
      try {
        F(g);
        var q = !1;
      } catch {
        q = !0;
      }
      if (q) {
        q = console;
        var K = q.error, I = typeof Symbol == "function" && Symbol.toStringTag && g[Symbol.toStringTag] || g.constructor.name || "Object";
        return K.call(
          q,
          "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
          I
        ), F(g);
      }
    }
    function _(g) {
      if (g === Mt) return "<>";
      if (typeof g == "object" && g !== null && g.$$typeof === Tt)
        return "<...>";
      try {
        var q = M(g);
        return q ? "<" + q + ">" : "<...>";
      } catch {
        return "<...>";
      }
    }
    function ie() {
      var g = Te.A;
      return g === null ? null : g.getOwner();
    }
    function he() {
      return Error("react-stack-top-frame");
    }
    function Oe(g) {
      if (Ce.call(g, "key")) {
        var q = Object.getOwnPropertyDescriptor(g, "key").get;
        if (q && q.isReactWarning) return !1;
      }
      return g.key !== void 0;
    }
    function Se(g, q) {
      function K() {
        pt || (pt = !0, console.error(
          "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
          q
        ));
      }
      K.isReactWarning = !0, Object.defineProperty(g, "key", {
        get: K,
        configurable: !0
      });
    }
    function N() {
      var g = M(this.type);
      return O[g] || (O[g] = !0, console.error(
        "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
      )), g = this.props.ref, g !== void 0 ? g : null;
    }
    function V(g, q, K, I, ce, ze, oe, il) {
      return K = ze.ref, g = {
        $$typeof: ae,
        type: g,
        key: q,
        props: ze,
        _owner: ce
      }, (K !== void 0 ? K : null) !== null ? Object.defineProperty(g, "ref", {
        enumerable: !1,
        get: N
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
    function le(g, q, K, I, ce, ze, oe, il) {
      var Ne = q.children;
      if (Ne !== void 0)
        if (I)
          if (_t(Ne)) {
            for (I = 0; I < Ne.length; I++)
              k(Ne[I]);
            Object.freeze && Object.freeze(Ne);
          } else
            console.error(
              "React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead."
            );
        else k(Ne);
      if (Ce.call(q, "key")) {
        Ne = M(g);
        var wt = Object.keys(q).filter(function(zn) {
          return zn !== "key";
        });
        I = 0 < wt.length ? "{key: someKey, " + wt.join(": ..., ") + ": ...}" : "{key: someKey}", be[Ne + I] || (wt = 0 < wt.length ? "{" + wt.join(": ..., ") + ": ...}" : "{}", console.error(
          `A props object containing a "key" prop is being spread into JSX:
  let props = %s;
  <%s {...props} />
React keys must be passed directly to JSX without using spread:
  let props = %s;
  <%s key={someKey} {...props} />`,
          I,
          Ne,
          wt,
          Ne
        ), be[Ne + I] = !0);
      }
      if (Ne = null, K !== void 0 && (re(K), Ne = "" + K), Oe(q) && (re(q.key), Ne = "" + q.key), "key" in q) {
        K = {};
        for (var na in q)
          na !== "key" && (K[na] = q[na]);
      } else K = q;
      return Ne && Se(
        K,
        typeof g == "function" ? g.displayName || g.name || "Unknown" : g
      ), V(
        g,
        Ne,
        ze,
        ce,
        ie(),
        K,
        oe,
        il
      );
    }
    function k(g) {
      typeof g == "object" && g !== null && g.$$typeof === ae && g._store && (g._store.validated = 1);
    }
    var U = xh(), ae = Symbol.for("react.transitional.element"), Ye = Symbol.for("react.portal"), Mt = Symbol.for("react.fragment"), $e = Symbol.for("react.strict_mode"), tt = Symbol.for("react.profiler"), Pt = Symbol.for("react.consumer"), Me = Symbol.for("react.context"), lt = Symbol.for("react.forward_ref"), De = Symbol.for("react.suspense"), bt = Symbol.for("react.suspense_list"), Ge = Symbol.for("react.memo"), Tt = Symbol.for("react.lazy"), de = Symbol.for("react.activity"), Rt = Symbol.for("react.client.reference"), Te = U.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, Ce = Object.prototype.hasOwnProperty, _t = Array.isArray, Gt = console.createTask ? console.createTask : function() {
      return null;
    };
    U = {
      react_stack_bottom_frame: function(g) {
        return g();
      }
    };
    var pt, O = {}, W = U.react_stack_bottom_frame.bind(
      U,
      he
    )(), P = Gt(_(he)), be = {};
    Ap.Fragment = Mt, Ap.jsx = function(g, q, K, I, ce) {
      var ze = 1e4 > Te.recentlyCreatedOwnerStacks++;
      return le(
        g,
        q,
        K,
        !1,
        I,
        ce,
        ze ? Error("react-stack-top-frame") : W,
        ze ? Gt(_(g)) : P
      );
    }, Ap.jsxs = function(g, q, K, I, ce) {
      var ze = 1e4 > Te.recentlyCreatedOwnerStacks++;
      return le(
        g,
        q,
        K,
        !0,
        I,
        ce,
        ze ? Error("react-stack-top-frame") : W,
        ze ? Gt(_(g)) : P
      );
    };
  }()), Ap;
}
var Jb;
function AT() {
  return Jb || (Jb = 1, It.env.NODE_ENV === "production" ? vg.exports = bT() : vg.exports = ET()), vg.exports;
}
var ge = AT(), Dn = xh(), bg = { exports: {} }, Rp = {}, Sg = { exports: {} }, G0 = {};
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
function RT() {
  return kb || (kb = 1, function(M) {
    function F(O, W) {
      var P = O.length;
      O.push(W);
      e: for (; 0 < P; ) {
        var be = P - 1 >>> 1, g = O[be];
        if (0 < ie(g, W))
          O[be] = W, O[P] = g, P = be;
        else break e;
      }
    }
    function re(O) {
      return O.length === 0 ? null : O[0];
    }
    function _(O) {
      if (O.length === 0) return null;
      var W = O[0], P = O.pop();
      if (P !== W) {
        O[0] = P;
        e: for (var be = 0, g = O.length, q = g >>> 1; be < q; ) {
          var K = 2 * (be + 1) - 1, I = O[K], ce = K + 1, ze = O[ce];
          if (0 > ie(I, P))
            ce < g && 0 > ie(ze, I) ? (O[be] = ze, O[ce] = P, be = ce) : (O[be] = I, O[K] = P, be = K);
          else if (ce < g && 0 > ie(ze, P))
            O[be] = ze, O[ce] = P, be = ce;
          else break e;
        }
      }
      return W;
    }
    function ie(O, W) {
      var P = O.sortIndex - W.sortIndex;
      return P !== 0 ? P : O.id - W.id;
    }
    if (M.unstable_now = void 0, typeof performance == "object" && typeof performance.now == "function") {
      var he = performance;
      M.unstable_now = function() {
        return he.now();
      };
    } else {
      var Oe = Date, Se = Oe.now();
      M.unstable_now = function() {
        return Oe.now() - Se;
      };
    }
    var N = [], V = [], le = 1, k = null, U = 3, ae = !1, Ye = !1, Mt = !1, $e = !1, tt = typeof setTimeout == "function" ? setTimeout : null, Pt = typeof clearTimeout == "function" ? clearTimeout : null, Me = typeof setImmediate < "u" ? setImmediate : null;
    function lt(O) {
      for (var W = re(V); W !== null; ) {
        if (W.callback === null) _(V);
        else if (W.startTime <= O)
          _(V), W.sortIndex = W.expirationTime, F(N, W);
        else break;
        W = re(V);
      }
    }
    function De(O) {
      if (Mt = !1, lt(O), !Ye)
        if (re(N) !== null)
          Ye = !0, bt || (bt = !0, Ce());
        else {
          var W = re(V);
          W !== null && pt(De, W.startTime - O);
        }
    }
    var bt = !1, Ge = -1, Tt = 5, de = -1;
    function Rt() {
      return $e ? !0 : !(M.unstable_now() - de < Tt);
    }
    function Te() {
      if ($e = !1, bt) {
        var O = M.unstable_now();
        de = O;
        var W = !0;
        try {
          e: {
            Ye = !1, Mt && (Mt = !1, Pt(Ge), Ge = -1), ae = !0;
            var P = U;
            try {
              t: {
                for (lt(O), k = re(N); k !== null && !(k.expirationTime > O && Rt()); ) {
                  var be = k.callback;
                  if (typeof be == "function") {
                    k.callback = null, U = k.priorityLevel;
                    var g = be(
                      k.expirationTime <= O
                    );
                    if (O = M.unstable_now(), typeof g == "function") {
                      k.callback = g, lt(O), W = !0;
                      break t;
                    }
                    k === re(N) && _(N), lt(O);
                  } else _(N);
                  k = re(N);
                }
                if (k !== null) W = !0;
                else {
                  var q = re(V);
                  q !== null && pt(
                    De,
                    q.startTime - O
                  ), W = !1;
                }
              }
              break e;
            } finally {
              k = null, U = P, ae = !1;
            }
            W = void 0;
          }
        } finally {
          W ? Ce() : bt = !1;
        }
      }
    }
    var Ce;
    if (typeof Me == "function")
      Ce = function() {
        Me(Te);
      };
    else if (typeof MessageChannel < "u") {
      var _t = new MessageChannel(), Gt = _t.port2;
      _t.port1.onmessage = Te, Ce = function() {
        Gt.postMessage(null);
      };
    } else
      Ce = function() {
        tt(Te, 0);
      };
    function pt(O, W) {
      Ge = tt(function() {
        O(M.unstable_now());
      }, W);
    }
    M.unstable_IdlePriority = 5, M.unstable_ImmediatePriority = 1, M.unstable_LowPriority = 4, M.unstable_NormalPriority = 3, M.unstable_Profiling = null, M.unstable_UserBlockingPriority = 2, M.unstable_cancelCallback = function(O) {
      O.callback = null;
    }, M.unstable_forceFrameRate = function(O) {
      0 > O || 125 < O ? console.error(
        "forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"
      ) : Tt = 0 < O ? Math.floor(1e3 / O) : 5;
    }, M.unstable_getCurrentPriorityLevel = function() {
      return U;
    }, M.unstable_next = function(O) {
      switch (U) {
        case 1:
        case 2:
        case 3:
          var W = 3;
          break;
        default:
          W = U;
      }
      var P = U;
      U = W;
      try {
        return O();
      } finally {
        U = P;
      }
    }, M.unstable_requestPaint = function() {
      $e = !0;
    }, M.unstable_runWithPriority = function(O, W) {
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
      var P = U;
      U = O;
      try {
        return W();
      } finally {
        U = P;
      }
    }, M.unstable_scheduleCallback = function(O, W, P) {
      var be = M.unstable_now();
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
        id: le++,
        callback: W,
        priorityLevel: O,
        startTime: P,
        expirationTime: g,
        sortIndex: -1
      }, P > be ? (O.sortIndex = P, F(V, O), re(N) === null && O === re(V) && (Mt ? (Pt(Ge), Ge = -1) : Mt = !0, pt(De, P - be))) : (O.sortIndex = g, F(N, O), Ye || ae || (Ye = !0, bt || (bt = !0, Ce()))), O;
    }, M.unstable_shouldYield = Rt, M.unstable_wrapCallback = function(O) {
      var W = U;
      return function() {
        var P = U;
        U = W;
        try {
          return O.apply(this, arguments);
        } finally {
          U = P;
        }
      };
    };
  }(G0)), G0;
}
var L0 = {}, $b;
function OT() {
  return $b || ($b = 1, function(M) {
    It.env.NODE_ENV !== "production" && function() {
      function F() {
        if (De = !1, de) {
          var O = M.unstable_now();
          Ce = O;
          var W = !0;
          try {
            e: {
              Me = !1, lt && (lt = !1, Ge(Rt), Rt = -1), Pt = !0;
              var P = tt;
              try {
                t: {
                  for (Oe(O), $e = _(ae); $e !== null && !($e.expirationTime > O && N()); ) {
                    var be = $e.callback;
                    if (typeof be == "function") {
                      $e.callback = null, tt = $e.priorityLevel;
                      var g = be(
                        $e.expirationTime <= O
                      );
                      if (O = M.unstable_now(), typeof g == "function") {
                        $e.callback = g, Oe(O), W = !0;
                        break t;
                      }
                      $e === _(ae) && ie(ae), Oe(O);
                    } else ie(ae);
                    $e = _(ae);
                  }
                  if ($e !== null) W = !0;
                  else {
                    var q = _(Ye);
                    q !== null && V(
                      Se,
                      q.startTime - O
                    ), W = !1;
                  }
                }
                break e;
              } finally {
                $e = null, tt = P, Pt = !1;
              }
              W = void 0;
            }
          } finally {
            W ? _t() : de = !1;
          }
        }
      }
      function re(O, W) {
        var P = O.length;
        O.push(W);
        e: for (; 0 < P; ) {
          var be = P - 1 >>> 1, g = O[be];
          if (0 < he(g, W))
            O[be] = W, O[P] = g, P = be;
          else break e;
        }
      }
      function _(O) {
        return O.length === 0 ? null : O[0];
      }
      function ie(O) {
        if (O.length === 0) return null;
        var W = O[0], P = O.pop();
        if (P !== W) {
          O[0] = P;
          e: for (var be = 0, g = O.length, q = g >>> 1; be < q; ) {
            var K = 2 * (be + 1) - 1, I = O[K], ce = K + 1, ze = O[ce];
            if (0 > he(I, P))
              ce < g && 0 > he(ze, I) ? (O[be] = ze, O[ce] = P, be = ce) : (O[be] = I, O[K] = P, be = K);
            else if (ce < g && 0 > he(ze, P))
              O[be] = ze, O[ce] = P, be = ce;
            else break e;
          }
        }
        return W;
      }
      function he(O, W) {
        var P = O.sortIndex - W.sortIndex;
        return P !== 0 ? P : O.id - W.id;
      }
      function Oe(O) {
        for (var W = _(Ye); W !== null; ) {
          if (W.callback === null) ie(Ye);
          else if (W.startTime <= O)
            ie(Ye), W.sortIndex = W.expirationTime, re(ae, W);
          else break;
          W = _(Ye);
        }
      }
      function Se(O) {
        if (lt = !1, Oe(O), !Me)
          if (_(ae) !== null)
            Me = !0, de || (de = !0, _t());
          else {
            var W = _(Ye);
            W !== null && V(
              Se,
              W.startTime - O
            );
          }
      }
      function N() {
        return De ? !0 : !(M.unstable_now() - Ce < Te);
      }
      function V(O, W) {
        Rt = bt(function() {
          O(M.unstable_now());
        }, W);
      }
      if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error()), M.unstable_now = void 0, typeof performance == "object" && typeof performance.now == "function") {
        var le = performance;
        M.unstable_now = function() {
          return le.now();
        };
      } else {
        var k = Date, U = k.now();
        M.unstable_now = function() {
          return k.now() - U;
        };
      }
      var ae = [], Ye = [], Mt = 1, $e = null, tt = 3, Pt = !1, Me = !1, lt = !1, De = !1, bt = typeof setTimeout == "function" ? setTimeout : null, Ge = typeof clearTimeout == "function" ? clearTimeout : null, Tt = typeof setImmediate < "u" ? setImmediate : null, de = !1, Rt = -1, Te = 5, Ce = -1;
      if (typeof Tt == "function")
        var _t = function() {
          Tt(F);
        };
      else if (typeof MessageChannel < "u") {
        var Gt = new MessageChannel(), pt = Gt.port2;
        Gt.port1.onmessage = F, _t = function() {
          pt.postMessage(null);
        };
      } else
        _t = function() {
          bt(F, 0);
        };
      M.unstable_IdlePriority = 5, M.unstable_ImmediatePriority = 1, M.unstable_LowPriority = 4, M.unstable_NormalPriority = 3, M.unstable_Profiling = null, M.unstable_UserBlockingPriority = 2, M.unstable_cancelCallback = function(O) {
        O.callback = null;
      }, M.unstable_forceFrameRate = function(O) {
        0 > O || 125 < O ? console.error(
          "forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"
        ) : Te = 0 < O ? Math.floor(1e3 / O) : 5;
      }, M.unstable_getCurrentPriorityLevel = function() {
        return tt;
      }, M.unstable_next = function(O) {
        switch (tt) {
          case 1:
          case 2:
          case 3:
            var W = 3;
            break;
          default:
            W = tt;
        }
        var P = tt;
        tt = W;
        try {
          return O();
        } finally {
          tt = P;
        }
      }, M.unstable_requestPaint = function() {
        De = !0;
      }, M.unstable_runWithPriority = function(O, W) {
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
        var P = tt;
        tt = O;
        try {
          return W();
        } finally {
          tt = P;
        }
      }, M.unstable_scheduleCallback = function(O, W, P) {
        var be = M.unstable_now();
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
          id: Mt++,
          callback: W,
          priorityLevel: O,
          startTime: P,
          expirationTime: g,
          sortIndex: -1
        }, P > be ? (O.sortIndex = P, re(Ye, O), _(ae) === null && O === _(Ye) && (lt ? (Ge(Rt), Rt = -1) : lt = !0, V(Se, P - be))) : (O.sortIndex = g, re(ae, O), Me || Pt || (Me = !0, de || (de = !0, _t()))), O;
      }, M.unstable_shouldYield = N, M.unstable_wrapCallback = function(O) {
        var W = tt;
        return function() {
          var P = tt;
          tt = W;
          try {
            return O.apply(this, arguments);
          } finally {
            tt = P;
          }
        };
      }, typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
    }();
  }(L0)), L0;
}
var Wb;
function cS() {
  return Wb || (Wb = 1, It.env.NODE_ENV === "production" ? Sg.exports = RT() : Sg.exports = OT()), Sg.exports;
}
var Tg = { exports: {} }, Ta = {};
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
  if (Fb) return Ta;
  Fb = 1;
  var M = xh();
  function F(N) {
    var V = "https://react.dev/errors/" + N;
    if (1 < arguments.length) {
      V += "?args[]=" + encodeURIComponent(arguments[1]);
      for (var le = 2; le < arguments.length; le++)
        V += "&args[]=" + encodeURIComponent(arguments[le]);
    }
    return "Minified React error #" + N + "; visit " + V + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  function re() {
  }
  var _ = {
    d: {
      f: re,
      r: function() {
        throw Error(F(522));
      },
      D: re,
      C: re,
      L: re,
      m: re,
      X: re,
      S: re,
      M: re
    },
    p: 0,
    findDOMNode: null
  }, ie = Symbol.for("react.portal");
  function he(N, V, le) {
    var k = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: ie,
      key: k == null ? null : "" + k,
      children: N,
      containerInfo: V,
      implementation: le
    };
  }
  var Oe = M.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  function Se(N, V) {
    if (N === "font") return "";
    if (typeof V == "string")
      return V === "use-credentials" ? V : "";
  }
  return Ta.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = _, Ta.createPortal = function(N, V) {
    var le = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!V || V.nodeType !== 1 && V.nodeType !== 9 && V.nodeType !== 11)
      throw Error(F(299));
    return he(N, V, null, le);
  }, Ta.flushSync = function(N) {
    var V = Oe.T, le = _.p;
    try {
      if (Oe.T = null, _.p = 2, N) return N();
    } finally {
      Oe.T = V, _.p = le, _.d.f();
    }
  }, Ta.preconnect = function(N, V) {
    typeof N == "string" && (V ? (V = V.crossOrigin, V = typeof V == "string" ? V === "use-credentials" ? V : "" : void 0) : V = null, _.d.C(N, V));
  }, Ta.prefetchDNS = function(N) {
    typeof N == "string" && _.d.D(N);
  }, Ta.preinit = function(N, V) {
    if (typeof N == "string" && V && typeof V.as == "string") {
      var le = V.as, k = Se(le, V.crossOrigin), U = typeof V.integrity == "string" ? V.integrity : void 0, ae = typeof V.fetchPriority == "string" ? V.fetchPriority : void 0;
      le === "style" ? _.d.S(
        N,
        typeof V.precedence == "string" ? V.precedence : void 0,
        {
          crossOrigin: k,
          integrity: U,
          fetchPriority: ae
        }
      ) : le === "script" && _.d.X(N, {
        crossOrigin: k,
        integrity: U,
        fetchPriority: ae,
        nonce: typeof V.nonce == "string" ? V.nonce : void 0
      });
    }
  }, Ta.preinitModule = function(N, V) {
    if (typeof N == "string")
      if (typeof V == "object" && V !== null) {
        if (V.as == null || V.as === "script") {
          var le = Se(
            V.as,
            V.crossOrigin
          );
          _.d.M(N, {
            crossOrigin: le,
            integrity: typeof V.integrity == "string" ? V.integrity : void 0,
            nonce: typeof V.nonce == "string" ? V.nonce : void 0
          });
        }
      } else V == null && _.d.M(N);
  }, Ta.preload = function(N, V) {
    if (typeof N == "string" && typeof V == "object" && V !== null && typeof V.as == "string") {
      var le = V.as, k = Se(le, V.crossOrigin);
      _.d.L(N, le, {
        crossOrigin: k,
        integrity: typeof V.integrity == "string" ? V.integrity : void 0,
        nonce: typeof V.nonce == "string" ? V.nonce : void 0,
        type: typeof V.type == "string" ? V.type : void 0,
        fetchPriority: typeof V.fetchPriority == "string" ? V.fetchPriority : void 0,
        referrerPolicy: typeof V.referrerPolicy == "string" ? V.referrerPolicy : void 0,
        imageSrcSet: typeof V.imageSrcSet == "string" ? V.imageSrcSet : void 0,
        imageSizes: typeof V.imageSizes == "string" ? V.imageSizes : void 0,
        media: typeof V.media == "string" ? V.media : void 0
      });
    }
  }, Ta.preloadModule = function(N, V) {
    if (typeof N == "string")
      if (V) {
        var le = Se(V.as, V.crossOrigin);
        _.d.m(N, {
          as: typeof V.as == "string" && V.as !== "script" ? V.as : void 0,
          crossOrigin: le,
          integrity: typeof V.integrity == "string" ? V.integrity : void 0
        });
      } else _.d.m(N);
  }, Ta.requestFormReset = function(N) {
    _.d.r(N);
  }, Ta.unstable_batchedUpdates = function(N, V) {
    return N(V);
  }, Ta.useFormState = function(N, V, le) {
    return Oe.H.useFormState(N, V, le);
  }, Ta.useFormStatus = function() {
    return Oe.H.useHostTransitionStatus();
  }, Ta.version = "19.1.1", Ta;
}
var Ea = {}, Ib;
function zT() {
  return Ib || (Ib = 1, It.env.NODE_ENV !== "production" && function() {
    function M() {
    }
    function F(k) {
      return "" + k;
    }
    function re(k, U, ae) {
      var Ye = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
      try {
        F(Ye);
        var Mt = !1;
      } catch {
        Mt = !0;
      }
      return Mt && (console.error(
        "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
        typeof Symbol == "function" && Symbol.toStringTag && Ye[Symbol.toStringTag] || Ye.constructor.name || "Object"
      ), F(Ye)), {
        $$typeof: V,
        key: Ye == null ? null : "" + Ye,
        children: k,
        containerInfo: U,
        implementation: ae
      };
    }
    function _(k, U) {
      if (k === "font") return "";
      if (typeof U == "string")
        return U === "use-credentials" ? U : "";
    }
    function ie(k) {
      return k === null ? "`null`" : k === void 0 ? "`undefined`" : k === "" ? "an empty string" : 'something with type "' + typeof k + '"';
    }
    function he(k) {
      return k === null ? "`null`" : k === void 0 ? "`undefined`" : k === "" ? "an empty string" : typeof k == "string" ? JSON.stringify(k) : typeof k == "number" ? "`" + k + "`" : 'something with type "' + typeof k + '"';
    }
    function Oe() {
      var k = le.H;
      return k === null && console.error(
        `Invalid hook call. Hooks can only be called inside of the body of a function component. This could happen for one of the following reasons:
1. You might have mismatching versions of React and the renderer (such as React DOM)
2. You might be breaking the Rules of Hooks
3. You might have more than one copy of React in the same app
See https://react.dev/link/invalid-hook-call for tips about how to debug and fix this problem.`
      ), k;
    }
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var Se = xh(), N = {
      d: {
        f: M,
        r: function() {
          throw Error(
            "Invalid form element. requestFormReset must be passed a form that was rendered by React."
          );
        },
        D: M,
        C: M,
        L: M,
        m: M,
        X: M,
        S: M,
        M
      },
      p: 0,
      findDOMNode: null
    }, V = Symbol.for("react.portal"), le = Se.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    typeof Map == "function" && Map.prototype != null && typeof Map.prototype.forEach == "function" && typeof Set == "function" && Set.prototype != null && typeof Set.prototype.clear == "function" && typeof Set.prototype.forEach == "function" || console.error(
      "React depends on Map and Set built-in types. Make sure that you load a polyfill in older browsers. https://reactjs.org/link/react-polyfills"
    ), Ea.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = N, Ea.createPortal = function(k, U) {
      var ae = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
      if (!U || U.nodeType !== 1 && U.nodeType !== 9 && U.nodeType !== 11)
        throw Error("Target container is not a DOM element.");
      return re(k, U, null, ae);
    }, Ea.flushSync = function(k) {
      var U = le.T, ae = N.p;
      try {
        if (le.T = null, N.p = 2, k)
          return k();
      } finally {
        le.T = U, N.p = ae, N.d.f() && console.error(
          "flushSync was called from inside a lifecycle method. React cannot flush when React is already rendering. Consider moving this call to a scheduler task or micro task."
        );
      }
    }, Ea.preconnect = function(k, U) {
      typeof k == "string" && k ? U != null && typeof U != "object" ? console.error(
        "ReactDOM.preconnect(): Expected the `options` argument (second) to be an object but encountered %s instead. The only supported option at this time is `crossOrigin` which accepts a string.",
        he(U)
      ) : U != null && typeof U.crossOrigin != "string" && console.error(
        "ReactDOM.preconnect(): Expected the `crossOrigin` option (second argument) to be a string but encountered %s instead. Try removing this option or passing a string value instead.",
        ie(U.crossOrigin)
      ) : console.error(
        "ReactDOM.preconnect(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
        ie(k)
      ), typeof k == "string" && (U ? (U = U.crossOrigin, U = typeof U == "string" ? U === "use-credentials" ? U : "" : void 0) : U = null, N.d.C(k, U));
    }, Ea.prefetchDNS = function(k) {
      if (typeof k != "string" || !k)
        console.error(
          "ReactDOM.prefetchDNS(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
          ie(k)
        );
      else if (1 < arguments.length) {
        var U = arguments[1];
        typeof U == "object" && U.hasOwnProperty("crossOrigin") ? console.error(
          "ReactDOM.prefetchDNS(): Expected only one argument, `href`, but encountered %s as a second argument instead. This argument is reserved for future options and is currently disallowed. It looks like the you are attempting to set a crossOrigin property for this DNS lookup hint. Browsers do not perform DNS queries using CORS and setting this attribute on the resource hint has no effect. Try calling ReactDOM.prefetchDNS() with just a single string argument, `href`.",
          he(U)
        ) : console.error(
          "ReactDOM.prefetchDNS(): Expected only one argument, `href`, but encountered %s as a second argument instead. This argument is reserved for future options and is currently disallowed. Try calling ReactDOM.prefetchDNS() with just a single string argument, `href`.",
          he(U)
        );
      }
      typeof k == "string" && N.d.D(k);
    }, Ea.preinit = function(k, U) {
      if (typeof k == "string" && k ? U == null || typeof U != "object" ? console.error(
        "ReactDOM.preinit(): Expected the `options` argument (second) to be an object with an `as` property describing the type of resource to be preinitialized but encountered %s instead.",
        he(U)
      ) : U.as !== "style" && U.as !== "script" && console.error(
        'ReactDOM.preinit(): Expected the `as` property in the `options` argument (second) to contain a valid value describing the type of resource to be preinitialized but encountered %s instead. Valid values for `as` are "style" and "script".',
        he(U.as)
      ) : console.error(
        "ReactDOM.preinit(): Expected the `href` argument (first) to be a non-empty string but encountered %s instead.",
        ie(k)
      ), typeof k == "string" && U && typeof U.as == "string") {
        var ae = U.as, Ye = _(ae, U.crossOrigin), Mt = typeof U.integrity == "string" ? U.integrity : void 0, $e = typeof U.fetchPriority == "string" ? U.fetchPriority : void 0;
        ae === "style" ? N.d.S(
          k,
          typeof U.precedence == "string" ? U.precedence : void 0,
          {
            crossOrigin: Ye,
            integrity: Mt,
            fetchPriority: $e
          }
        ) : ae === "script" && N.d.X(k, {
          crossOrigin: Ye,
          integrity: Mt,
          fetchPriority: $e,
          nonce: typeof U.nonce == "string" ? U.nonce : void 0
        });
      }
    }, Ea.preinitModule = function(k, U) {
      var ae = "";
      if (typeof k == "string" && k || (ae += " The `href` argument encountered was " + ie(k) + "."), U !== void 0 && typeof U != "object" ? ae += " The `options` argument encountered was " + ie(U) + "." : U && "as" in U && U.as !== "script" && (ae += " The `as` option encountered was " + he(U.as) + "."), ae)
        console.error(
          "ReactDOM.preinitModule(): Expected up to two arguments, a non-empty `href` string and, optionally, an `options` object with a valid `as` property.%s",
          ae
        );
      else
        switch (ae = U && typeof U.as == "string" ? U.as : "script", ae) {
          case "script":
            break;
          default:
            ae = he(ae), console.error(
              'ReactDOM.preinitModule(): Currently the only supported "as" type for this function is "script" but received "%s" instead. This warning was generated for `href` "%s". In the future other module types will be supported, aligning with the import-attributes proposal. Learn more here: (https://github.com/tc39/proposal-import-attributes)',
              ae,
              k
            );
        }
      typeof k == "string" && (typeof U == "object" && U !== null ? (U.as == null || U.as === "script") && (ae = _(
        U.as,
        U.crossOrigin
      ), N.d.M(k, {
        crossOrigin: ae,
        integrity: typeof U.integrity == "string" ? U.integrity : void 0,
        nonce: typeof U.nonce == "string" ? U.nonce : void 0
      })) : U == null && N.d.M(k));
    }, Ea.preload = function(k, U) {
      var ae = "";
      if (typeof k == "string" && k || (ae += " The `href` argument encountered was " + ie(k) + "."), U == null || typeof U != "object" ? ae += " The `options` argument encountered was " + ie(U) + "." : typeof U.as == "string" && U.as || (ae += " The `as` option encountered was " + ie(U.as) + "."), ae && console.error(
        'ReactDOM.preload(): Expected two arguments, a non-empty `href` string and an `options` object with an `as` property valid for a `<link rel="preload" as="..." />` tag.%s',
        ae
      ), typeof k == "string" && typeof U == "object" && U !== null && typeof U.as == "string") {
        ae = U.as;
        var Ye = _(
          ae,
          U.crossOrigin
        );
        N.d.L(k, ae, {
          crossOrigin: Ye,
          integrity: typeof U.integrity == "string" ? U.integrity : void 0,
          nonce: typeof U.nonce == "string" ? U.nonce : void 0,
          type: typeof U.type == "string" ? U.type : void 0,
          fetchPriority: typeof U.fetchPriority == "string" ? U.fetchPriority : void 0,
          referrerPolicy: typeof U.referrerPolicy == "string" ? U.referrerPolicy : void 0,
          imageSrcSet: typeof U.imageSrcSet == "string" ? U.imageSrcSet : void 0,
          imageSizes: typeof U.imageSizes == "string" ? U.imageSizes : void 0,
          media: typeof U.media == "string" ? U.media : void 0
        });
      }
    }, Ea.preloadModule = function(k, U) {
      var ae = "";
      typeof k == "string" && k || (ae += " The `href` argument encountered was " + ie(k) + "."), U !== void 0 && typeof U != "object" ? ae += " The `options` argument encountered was " + ie(U) + "." : U && "as" in U && typeof U.as != "string" && (ae += " The `as` option encountered was " + ie(U.as) + "."), ae && console.error(
        'ReactDOM.preloadModule(): Expected two arguments, a non-empty `href` string and, optionally, an `options` object with an `as` property valid for a `<link rel="modulepreload" as="..." />` tag.%s',
        ae
      ), typeof k == "string" && (U ? (ae = _(
        U.as,
        U.crossOrigin
      ), N.d.m(k, {
        as: typeof U.as == "string" && U.as !== "script" ? U.as : void 0,
        crossOrigin: ae,
        integrity: typeof U.integrity == "string" ? U.integrity : void 0
      })) : N.d.m(k));
    }, Ea.requestFormReset = function(k) {
      N.d.r(k);
    }, Ea.unstable_batchedUpdates = function(k, U) {
      return k(U);
    }, Ea.useFormState = function(k, U, ae) {
      return Oe().useFormState(k, U, ae);
    }, Ea.useFormStatus = function() {
      return Oe().useHostTransitionStatus();
    }, Ea.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  }()), Ea;
}
var Pb;
function oS() {
  if (Pb) return Tg.exports;
  Pb = 1;
  function M() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) {
      if (It.env.NODE_ENV !== "production")
        throw new Error("^_^");
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(M);
      } catch (F) {
        console.error(F);
      }
    }
  }
  return It.env.NODE_ENV === "production" ? (M(), Tg.exports = DT()) : Tg.exports = zT(), Tg.exports;
}
var eS;
function MT() {
  if (eS) return Rp;
  eS = 1;
  var M = cS(), F = xh(), re = oS();
  function _(l) {
    var n = "https://react.dev/errors/" + l;
    if (1 < arguments.length) {
      n += "?args[]=" + encodeURIComponent(arguments[1]);
      for (var u = 2; u < arguments.length; u++)
        n += "&args[]=" + encodeURIComponent(arguments[u]);
    }
    return "Minified React error #" + l + "; visit " + n + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  function ie(l) {
    return !(!l || l.nodeType !== 1 && l.nodeType !== 9 && l.nodeType !== 11);
  }
  function he(l) {
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
  function Oe(l) {
    if (l.tag === 13) {
      var n = l.memoizedState;
      if (n === null && (l = l.alternate, l !== null && (n = l.memoizedState)), n !== null) return n.dehydrated;
    }
    return null;
  }
  function Se(l) {
    if (he(l) !== l)
      throw Error(_(188));
  }
  function N(l) {
    var n = l.alternate;
    if (!n) {
      if (n = he(l), n === null) throw Error(_(188));
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
          if (s === u) return Se(r), l;
          if (s === c) return Se(r), n;
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
  function V(l) {
    var n = l.tag;
    if (n === 5 || n === 26 || n === 27 || n === 6) return l;
    for (l = l.child; l !== null; ) {
      if (n = V(l), n !== null) return n;
      l = l.sibling;
    }
    return null;
  }
  var le = Object.assign, k = Symbol.for("react.element"), U = Symbol.for("react.transitional.element"), ae = Symbol.for("react.portal"), Ye = Symbol.for("react.fragment"), Mt = Symbol.for("react.strict_mode"), $e = Symbol.for("react.profiler"), tt = Symbol.for("react.provider"), Pt = Symbol.for("react.consumer"), Me = Symbol.for("react.context"), lt = Symbol.for("react.forward_ref"), De = Symbol.for("react.suspense"), bt = Symbol.for("react.suspense_list"), Ge = Symbol.for("react.memo"), Tt = Symbol.for("react.lazy"), de = Symbol.for("react.activity"), Rt = Symbol.for("react.memo_cache_sentinel"), Te = Symbol.iterator;
  function Ce(l) {
    return l === null || typeof l != "object" ? null : (l = Te && l[Te] || l["@@iterator"], typeof l == "function" ? l : null);
  }
  var _t = Symbol.for("react.client.reference");
  function Gt(l) {
    if (l == null) return null;
    if (typeof l == "function")
      return l.$$typeof === _t ? null : l.displayName || l.name || null;
    if (typeof l == "string") return l;
    switch (l) {
      case Ye:
        return "Fragment";
      case $e:
        return "Profiler";
      case Mt:
        return "StrictMode";
      case De:
        return "Suspense";
      case bt:
        return "SuspenseList";
      case de:
        return "Activity";
    }
    if (typeof l == "object")
      switch (l.$$typeof) {
        case ae:
          return "Portal";
        case Me:
          return (l.displayName || "Context") + ".Provider";
        case Pt:
          return (l._context.displayName || "Context") + ".Consumer";
        case lt:
          var n = l.render;
          return l = l.displayName, l || (l = n.displayName || n.name || "", l = l !== "" ? "ForwardRef(" + l + ")" : "ForwardRef"), l;
        case Ge:
          return n = l.displayName || null, n !== null ? n : Gt(l.type) || "Memo";
        case Tt:
          n = l._payload, l = l._init;
          try {
            return Gt(l(n));
          } catch {
          }
      }
    return null;
  }
  var pt = Array.isArray, O = F.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, W = re.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, P = {
    pending: !1,
    data: null,
    method: null,
    action: null
  }, be = [], g = -1;
  function q(l) {
    return { current: l };
  }
  function K(l) {
    0 > g || (l.current = be[g], be[g] = null, g--);
  }
  function I(l, n) {
    g++, be[g] = l.current, l.current = n;
  }
  var ce = q(null), ze = q(null), oe = q(null), il = q(null);
  function Ne(l, n) {
    switch (I(oe, n), I(ze, l), I(ce, null), n.nodeType) {
      case 9:
      case 11:
        l = (l = n.documentElement) && (l = l.namespaceURI) ? Lu(l) : 0;
        break;
      default:
        if (l = n.tagName, n = n.namespaceURI)
          n = Lu(n), l = ko(n, l);
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
    K(ce), I(ce, l);
  }
  function wt() {
    K(ce), K(ze), K(oe);
  }
  function na(l) {
    l.memoizedState !== null && I(il, l);
    var n = ce.current, u = ko(n, l.type);
    n !== u && (I(ze, l), I(ce, u));
  }
  function zn(l) {
    ze.current === l && (K(ce), K(ze)), il.current === l && (K(il), ba._currentValue = P);
  }
  var Zi = Object.prototype.hasOwnProperty, Mn = M.unstable_scheduleCallback, Pc = M.unstable_cancelCallback, Sf = M.unstable_shouldYield, tl = M.unstable_requestPaint, pl = M.unstable_now, Iu = M.unstable_getCurrentPriorityLevel, cs = M.unstable_ImmediatePriority, Je = M.unstable_UserBlockingPriority, Un = M.unstable_NormalPriority, eo = M.unstable_LowPriority, bu = M.unstable_IdlePriority, os = M.log, Tf = M.unstable_setDisableYieldValue, Pu = null, Ol = null;
  function Ya(l) {
    if (typeof os == "function" && Tf(l), Ol && typeof Ol.setStrictMode == "function")
      try {
        Ol.setStrictMode(Pu, l);
      } catch {
      }
  }
  var Dl = Math.clz32 ? Math.clz32 : lo, to = Math.log, Ef = Math.LN2;
  function lo(l) {
    return l >>>= 0, l === 0 ? 32 : 31 - (to(l) / Ef | 0) | 0;
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
    var r = 0, s = l.suspendedLanes, y = l.pingedLanes;
    l = l.warmLanes;
    var p = c & 134217727;
    return p !== 0 ? (c = p & ~s, c !== 0 ? r = zl(c) : (y &= p, y !== 0 ? r = zl(y) : u || (u = p & ~l, u !== 0 && (r = zl(u))))) : (p = c & ~s, p !== 0 ? r = zl(p) : y !== 0 ? r = zl(y) : u || (u = c & ~l, u !== 0 && (r = zl(u)))), r === 0 ? 0 : n !== 0 && n !== r && (n & s) === 0 && (s = r & -r, u = n & -n, s >= u || s === 32 && (u & 4194048) !== 0) ? n : r;
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
  function te() {
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
  function we(l, n) {
    l.pendingLanes |= n, n !== 268435456 && (l.suspendedLanes = 0, l.pingedLanes = 0, l.warmLanes = 0);
  }
  function Le(l, n, u, c, r, s) {
    var y = l.pendingLanes;
    l.pendingLanes = u, l.suspendedLanes = 0, l.pingedLanes = 0, l.warmLanes = 0, l.expiredLanes &= u, l.entangledLanes &= u, l.errorRecoveryDisabledLanes &= u, l.shellSuspendCounter = 0;
    var p = l.entanglements, S = l.expirationTimes, x = l.hiddenUpdates;
    for (u = y & ~u; 0 < u; ) {
      var Z = 31 - Dl(u), $ = 1 << Z;
      p[Z] = 0, S[Z] = -1;
      var w = x[Z];
      if (w !== null)
        for (x[Z] = null, Z = 0; Z < w.length; Z++) {
          var Y = w[Z];
          Y !== null && (Y.lane &= -536870913);
        }
      u &= ~$;
    }
    c !== 0 && ct(l, c, 0), s !== 0 && r === 0 && l.tag !== 0 && (l.suspendedLanes |= s & ~(y & ~n));
  }
  function ct(l, n, u) {
    l.pendingLanes |= n, l.suspendedLanes &= ~n;
    var c = 31 - Dl(n);
    l.entangledLanes |= n, l.entanglements[c] = l.entanglements[c] | 1073741824 | u & 4194090;
  }
  function je(l, n) {
    var u = l.entangledLanes |= n;
    for (l = l.entanglements; u; ) {
      var c = 31 - Dl(u), r = 1 << c;
      r & n | l[c] & n && (l[c] |= n), u &= ~r;
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
  function fs() {
    var l = W.p;
    return l !== 0 ? l : (l = window.event, l === void 0 ? 32 : Bm(l.type));
  }
  function Hh(l, n) {
    var u = W.p;
    try {
      return W.p = l, n();
    } finally {
      W.p = u;
    }
  }
  var cl = Math.random().toString(36).slice(2), vl = "__reactFiber$" + cl, kl = "__reactProps$" + cl, ao = "__reactContainer$" + cl, rs = "__reactEvents$" + cl, zp = "__reactListeners$" + cl, ss = "__reactHandles$" + cl, Mp = "__reactResources$" + cl, ye = "__reactMarker$" + cl;
  function Af(l) {
    delete l[vl], delete l[kl], delete l[rs], delete l[zp], delete l[ss];
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
  function Rf(l) {
    var n = l.tag;
    if (n === 5 || n === 26 || n === 27 || n === 6) return l.stateNode;
    throw Error(_(33));
  }
  function Su(l) {
    var n = l[Mp];
    return n || (n = l[Mp] = { hoistableStyles: /* @__PURE__ */ new Map(), hoistableScripts: /* @__PURE__ */ new Map() }), n;
  }
  function ol(l) {
    l[ye] = !0;
  }
  var Of = /* @__PURE__ */ new Set(), Aa = {};
  function ei(l, n) {
    ti(l, n), ti(l + "Capture", n);
  }
  function ti(l, n) {
    for (Aa[l] = n, l = 0; l < n.length; l++)
      Of.add(n[l]);
  }
  var Up = RegExp(
    "^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
  ), ds = {}, Nh = {};
  function _p(l) {
    return Zi.call(Nh, l) ? !0 : Zi.call(ds, l) ? !1 : Up.test(l) ? Nh[l] = !0 : (ds[l] = !0, !1);
  }
  function Tu(l, n, u) {
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
  var hs, wh;
  function Ji(l) {
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
                  var w = Y;
                }
                Reflect.construct(l, [], $);
              } else {
                try {
                  $.call();
                } catch (Y) {
                  w = Y;
                }
                l.call($.prototype);
              }
            } else {
              try {
                throw Error();
              } catch (Y) {
                w = Y;
              }
              ($ = l()) && typeof $.catch == "function" && $.catch(function() {
              });
            }
          } catch (Y) {
            if (Y && w && typeof Y.stack == "string")
              return [Y.stack, w.stack];
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
`), x = p.split(`
`);
        for (r = c = 0; c < S.length && !S[c].includes("DetermineComponentFrameRoot"); )
          c++;
        for (; r < x.length && !x[r].includes(
          "DetermineComponentFrameRoot"
        ); )
          r++;
        if (c === S.length || r === x.length)
          for (c = S.length - 1, r = x.length - 1; 1 <= c && 0 <= r && S[c] !== x[r]; )
            r--;
        for (; 1 <= c && 0 <= r; c--, r--)
          if (S[c] !== x[r]) {
            if (c !== 1 || r !== 1)
              do
                if (c--, r--, 0 > r || S[c] !== x[r]) {
                  var Z = `
` + S[c].replace(" at new ", " at ");
                  return l.displayName && Z.includes("<anonymous>") && (Z = Z.replace("<anonymous>", l.displayName)), Z;
                }
              while (1 <= c && 0 <= r);
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
  function qh(l) {
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
  function ai(l) {
    l._valueTracker || (l._valueTracker = Bh(l));
  }
  function $i(l) {
    if (!l) return !1;
    var n = l._valueTracker;
    if (!n) return !0;
    var u = n.getValue(), c = "";
    return l && (c = zf(l) ? l.checked ? "true" : "false" : l.value), l = c, l !== u ? (n.setValue(l), !0) : !1;
  }
  function no(l) {
    if (l = l || (typeof document < "u" ? document : void 0), typeof l > "u") return null;
    try {
      return l.activeElement || l.body;
    } catch {
      return l.body;
    }
  }
  var Ag = /[\n"\\]/g;
  function ja(l) {
    return l.replace(
      Ag,
      function(n) {
        return "\\" + n.charCodeAt(0).toString(16) + " ";
      }
    );
  }
  function ys(l, n, u, c, r, s, y, p) {
    l.name = "", y != null && typeof y != "function" && typeof y != "symbol" && typeof y != "boolean" ? l.type = y : l.removeAttribute("type"), n != null ? y === "number" ? (n === 0 && l.value === "" || l.value != n) && (l.value = "" + Gl(n)) : l.value !== "" + Gl(n) && (l.value = "" + Gl(n)) : y !== "submit" && y !== "reset" || l.removeAttribute("value"), n != null ? Mf(l, y, Gl(n)) : u != null ? Mf(l, y, Gl(u)) : c != null && l.removeAttribute("value"), r == null && s != null && (l.defaultChecked = !!s), r != null && (l.checked = r && typeof r != "function" && typeof r != "symbol"), p != null && typeof p != "function" && typeof p != "symbol" && typeof p != "boolean" ? l.name = "" + Gl(p) : l.removeAttribute("name");
  }
  function ms(l, n, u, c, r, s, y, p) {
    if (s != null && typeof s != "function" && typeof s != "symbol" && typeof s != "boolean" && (l.type = s), n != null || u != null) {
      if (!(s !== "submit" && s !== "reset" || n != null))
        return;
      u = u != null ? "" + Gl(u) : "", n = n != null ? "" + Gl(n) : u, p || n === l.value || (l.value = n), l.defaultValue = n;
    }
    c = c ?? r, c = typeof c != "function" && typeof c != "symbol" && !!c, l.checked = p ? l.checked : !!c, l.defaultChecked = !!c, y != null && typeof y != "function" && typeof y != "symbol" && typeof y != "boolean" && (l.name = y);
  }
  function Mf(l, n, u) {
    n === "number" && no(l.ownerDocument) === l || l.defaultValue === "" + u || (l.defaultValue = "" + u);
  }
  function Wi(l, n, u, c) {
    if (l = l.options, n) {
      n = {};
      for (var r = 0; r < u.length; r++)
        n["$" + u[r]] = !0;
      for (u = 0; u < l.length; u++)
        r = n.hasOwnProperty("$" + l[u].value), l[u].selected !== r && (l[u].selected = r), r && c && (l[u].defaultSelected = !0);
    } else {
      for (u = "" + Gl(u), n = null, r = 0; r < l.length; r++) {
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
    if (n != null && (n = "" + Gl(n), n !== l.value && (l.value = n), u == null)) {
      l.defaultValue !== n && (l.defaultValue = n);
      return;
    }
    l.defaultValue = u != null ? "" + Gl(u) : "";
  }
  function jh(l, n, u, c) {
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
  var Cp = new Set(
    "animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(
      " "
    )
  );
  function ps(l, n, u) {
    var c = n.indexOf("--") === 0;
    u == null || typeof u == "boolean" || u === "" ? c ? l.setProperty(n, "") : n === "float" ? l.cssFloat = "" : l[n] = "" : c ? l.setProperty(n, u) : typeof u != "number" || u === 0 || Cp.has(n) ? n === "float" ? l.cssFloat = u : l[n] = ("" + u).trim() : l[n] = u + "px";
  }
  function Uf(l, n, u) {
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
  var Rg = /* @__PURE__ */ new Map([
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
  function _f(l) {
    return xp.test("" + l) ? "javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')" : l;
  }
  var Ii = null;
  function vs(l) {
    return l = l.target || l.srcElement || window, l.correspondingUseElement && (l = l.correspondingUseElement), l.nodeType === 3 ? l.parentNode : l;
  }
  var io = null, co = null;
  function Hp(l) {
    var n = Ki(l);
    if (n && (l = n.stateNode)) {
      var u = l[kl] || null;
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
              'input[name="' + ja(
                "" + n
              ) + '"][type="radio"]'
            ), n = 0; n < u.length; n++) {
              var c = u[n];
              if (c !== l && c.form === l.form) {
                var r = c[kl] || null;
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
              c = u[n], c.form === l.form && $i(c);
          }
          break e;
        case "textarea":
          Yh(l, u.value, u.defaultValue);
          break e;
        case "select":
          n = u.value, n != null && Wi(l, !!u.multiple, n, !1);
      }
    }
  }
  var Gh = !1;
  function oo(l, n, u) {
    if (Gh) return l(n, u);
    Gh = !0;
    try {
      var c = l(n);
      return c;
    } finally {
      if (Gh = !1, (io !== null || co !== null) && (_c(), io && (n = io, l = co, co = io = null, Hp(n), l)))
        for (n = 0; n < l.length; n++) Hp(l[n]);
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
  var Cn = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), gs = !1;
  if (Cn)
    try {
      var Eu = {};
      Object.defineProperty(Eu, "passive", {
        get: function() {
          gs = !0;
        }
      }), window.addEventListener("test", Eu, Eu), window.removeEventListener("test", Eu, Eu);
    } catch {
      gs = !1;
    }
  var Au = null, fo = null, ec = null;
  function Lh() {
    if (ec) return ec;
    var l, n = fo, u = n.length, c, r = "value" in Au ? Au.value : Au.textContent, s = r.length;
    for (l = 0; l < u && n[l] === r[l]; l++) ;
    var y = u - l;
    for (c = 1; c <= y && n[u - c] === r[s - c]; c++) ;
    return ec = r.slice(l, 1 < c ? 1 - c : void 0);
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
  function Wl(l) {
    function n(u, c, r, s, y) {
      this._reactName = u, this._targetInst = r, this.type = c, this.nativeEvent = s, this.target = y, this.currentTarget = null;
      for (var p in l)
        l.hasOwnProperty(p) && (u = l[p], this[p] = u ? u(s) : s[p]);
      return this.isDefaultPrevented = (s.defaultPrevented != null ? s.defaultPrevented : s.returnValue === !1) ? bs : Ss, this.isPropagationStopped = Ss, this;
    }
    return le(n.prototype, {
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
  var ni = {
    eventPhase: 0,
    bubbles: 0,
    cancelable: 0,
    timeStamp: function(l) {
      return l.timeStamp || Date.now();
    },
    defaultPrevented: 0,
    isTrusted: 0
  }, Ts = Wl(ni), Cf = le({}, ni, { view: 0, detail: 0 }), Np = Wl(Cf), Vh, Es, xf, tc = le({}, Cf, {
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
    getModifierState: Ru,
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
  }), Xh = Wl(tc), wp = le({}, tc, { dataTransfer: 0 }), qp = Wl(wp), Og = le({}, Cf, { relatedTarget: 0 }), Qh = Wl(Og), Dg = le({}, ni, {
    animationName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), zg = Wl(Dg), Mg = le({}, ni, {
    clipboardData: function(l) {
      return "clipboardData" in l ? l.clipboardData : window.clipboardData;
    }
  }), Hf = Wl(Mg), Bp = le({}, ni, { data: 0 }), Zh = Wl(Bp), Yp = {
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
  function Ru() {
    return Gp;
  }
  var lc = le({}, Cf, {
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
    getModifierState: Ru,
    charCode: function(l) {
      return l.type === "keypress" ? Ul(l) : 0;
    },
    keyCode: function(l) {
      return l.type === "keydown" || l.type === "keyup" ? l.keyCode : 0;
    },
    which: function(l) {
      return l.type === "keypress" ? Ul(l) : l.type === "keydown" || l.type === "keyup" ? l.keyCode : 0;
    }
  }), on = Wl(lc), Ra = le({}, tc, {
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
  }), Nf = Wl(Ra), As = le({}, Cf, {
    touches: 0,
    targetTouches: 0,
    changedTouches: 0,
    altKey: 0,
    metaKey: 0,
    ctrlKey: 0,
    shiftKey: 0,
    getModifierState: Ru
  }), Jh = Wl(As), ia = le({}, ni, {
    propertyName: 0,
    elapsedTime: 0,
    pseudoElement: 0
  }), Lp = Wl(ia), Rs = le({}, tc, {
    deltaX: function(l) {
      return "deltaX" in l ? l.deltaX : "wheelDeltaX" in l ? -l.wheelDeltaX : 0;
    },
    deltaY: function(l) {
      return "deltaY" in l ? l.deltaY : "wheelDeltaY" in l ? -l.wheelDeltaY : "wheelDelta" in l ? -l.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), ac = Wl(Rs), kh = le({}, ni, {
    newState: 0,
    oldState: 0
  }), Vp = Wl(kh), Xp = [9, 13, 27, 32], wf = Cn && "CompositionEvent" in window, qf = null;
  Cn && "documentMode" in document && (qf = document.documentMode);
  var $h = Cn && "TextEvent" in window && !qf, xn = Cn && (!wf || qf && 8 < qf && 11 >= qf), Wh = " ", Os = !1;
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
  function ui(l) {
    return l = l.detail, typeof l == "object" && "data" in l ? l.data : null;
  }
  var ii = !1;
  function Fh(l, n) {
    switch (l) {
      case "compositionend":
        return ui(n);
      case "keypress":
        return n.which !== 32 ? null : (Os = !0, Wh);
      case "textInput":
        return l = n.data, l === Wh && Os ? null : l;
      default:
        return null;
    }
  }
  function nc(l, n) {
    if (ii)
      return l === "compositionend" || !wf && Bf(l, n) ? (l = Lh(), ec = fo = Au = null, ii = !1, l) : null;
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
        return xn && n.locale !== "ko" ? null : n.data;
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
    io ? co ? co.push(c) : co = [c] : io = c, n = Jo(n, "onChange"), 0 < n.length && (u = new Ts(
      "onChange",
      "change",
      null,
      u,
      c
    ), l.push({ event: u, listeners: n }));
  }
  var fn = null, rn = null;
  function Ih(l) {
    Nc(l, 0);
  }
  function Hn(l) {
    var n = Rf(l);
    if ($i(n)) return l;
  }
  function Ph(l, n) {
    if (l === "change") return n;
  }
  var ey = !1;
  if (Cn) {
    var uc;
    if (Cn) {
      var ic = "oninput" in document;
      if (!ic) {
        var ty = document.createElement("div");
        ty.setAttribute("oninput", "return;"), ic = typeof ty.oninput == "function";
      }
      uc = ic;
    } else uc = !1;
    ey = uc && (!document.documentMode || 9 < document.documentMode);
  }
  function ro() {
    fn && (fn.detachEvent("onpropertychange", ly), rn = fn = null);
  }
  function ly(l) {
    if (l.propertyName === "value" && Hn(rn)) {
      var n = [];
      zs(
        n,
        rn,
        l,
        vs(l)
      ), oo(Ih, n);
    }
  }
  function Ms(l, n, u) {
    l === "focusin" ? (ro(), fn = n, rn = u, fn.attachEvent("onpropertychange", ly)) : l === "focusout" && ro();
  }
  function ci(l) {
    if (l === "selectionchange" || l === "keyup" || l === "keydown")
      return Hn(rn);
  }
  function Ou(l, n) {
    if (l === "click") return Hn(n);
  }
  function ay(l, n) {
    if (l === "input" || l === "change")
      return Hn(n);
  }
  function ny(l, n) {
    return l === n && (l !== 0 || 1 / l === 1 / n) || l !== l && n !== n;
  }
  var _l = typeof Object.is == "function" ? Object.is : ny;
  function oi(l, n) {
    if (_l(l, n)) return !0;
    if (typeof l != "object" || l === null || typeof n != "object" || n === null)
      return !1;
    var u = Object.keys(l), c = Object.keys(n);
    if (u.length !== c.length) return !1;
    for (c = 0; c < u.length; c++) {
      var r = u[c];
      if (!Zi.call(n, r) || !_l(l[r], n[r]))
        return !1;
    }
    return !0;
  }
  function fi(l) {
    for (; l && l.firstChild; ) l = l.firstChild;
    return l;
  }
  function xt(l, n) {
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
  function Yf(l, n) {
    return l && n ? l === n ? !0 : l && l.nodeType === 3 ? !1 : n && n.nodeType === 3 ? Yf(l, n.parentNode) : "contains" in l ? l.contains(n) : l.compareDocumentPosition ? !!(l.compareDocumentPosition(n) & 16) : !1 : !1;
  }
  function uy(l) {
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
  function jf(l) {
    var n = l && l.nodeName && l.nodeName.toLowerCase();
    return n && (n === "input" && (l.type === "text" || l.type === "search" || l.type === "tel" || l.type === "url" || l.type === "password") || n === "textarea" || l.contentEditable === "true");
  }
  var cc = Cn && "documentMode" in document && 11 >= document.documentMode, Nn = null, sn = null, ri = null, oc = !1;
  function Us(l, n, u) {
    var c = u.window === u ? u.document : u.nodeType === 9 ? u : u.ownerDocument;
    oc || Nn == null || Nn !== no(c) || (c = Nn, "selectionStart" in c && jf(c) ? c = { start: c.selectionStart, end: c.selectionEnd } : (c = (c.ownerDocument && c.ownerDocument.defaultView || window).getSelection(), c = {
      anchorNode: c.anchorNode,
      anchorOffset: c.anchorOffset,
      focusNode: c.focusNode,
      focusOffset: c.focusOffset
    }), ri && oi(ri, c) || (ri = c, c = Jo(sn, "onSelect"), 0 < c.length && (n = new Ts(
      "onSelect",
      "select",
      null,
      n,
      u
    ), l.push({ event: n, listeners: c }), n.target = Nn)));
  }
  function Du(l, n) {
    var u = {};
    return u[l.toLowerCase()] = n.toLowerCase(), u["Webkit" + l] = "webkit" + n, u["Moz" + l] = "moz" + n, u;
  }
  var fc = {
    animationend: Du("Animation", "AnimationEnd"),
    animationiteration: Du("Animation", "AnimationIteration"),
    animationstart: Du("Animation", "AnimationStart"),
    transitionrun: Du("Transition", "TransitionRun"),
    transitionstart: Du("Transition", "TransitionStart"),
    transitioncancel: Du("Transition", "TransitionCancel"),
    transitionend: Du("Transition", "TransitionEnd")
  }, Ga = {}, dn = {};
  Cn && (dn = document.createElement("div").style, "AnimationEvent" in window || (delete fc.animationend.animation, delete fc.animationiteration.animation, delete fc.animationstart.animation), "TransitionEvent" in window || delete fc.transitionend.transition);
  function wn(l) {
    if (Ga[l]) return Ga[l];
    if (!fc[l]) return l;
    var n = fc[l], u;
    for (u in n)
      if (n.hasOwnProperty(u) && u in dn)
        return Ga[l] = n[u];
    return l;
  }
  var Zp = wn("animationend"), iy = wn("animationiteration"), Kp = wn("animationstart"), cy = wn("transitionrun"), _s = wn("transitionstart"), Jp = wn("transitioncancel"), oy = wn("transitionend"), fy = /* @__PURE__ */ new Map(), so = "abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(
    " "
  );
  so.push("scrollEnd");
  function La(l, n) {
    fy.set(l, n), ei(n, [l]);
  }
  var ry = /* @__PURE__ */ new WeakMap();
  function Oa(l, n) {
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
  var ca = [], si = 0, qn = 0;
  function hn() {
    for (var l = si, n = qn = si = 0; n < l; ) {
      var u = ca[n];
      ca[n++] = null;
      var c = ca[n];
      ca[n++] = null;
      var r = ca[n];
      ca[n++] = null;
      var s = ca[n];
      if (ca[n++] = null, c !== null && r !== null) {
        var y = c.pending;
        y === null ? r.next = r : (r.next = y.next, y.next = r), c.pending = r;
      }
      s !== 0 && yo(u, r, s);
    }
  }
  function di(l, n, u, c) {
    ca[si++] = l, ca[si++] = n, ca[si++] = u, ca[si++] = c, qn |= c, l.lanes |= c, l = l.alternate, l !== null && (l.lanes |= c);
  }
  function ho(l, n, u, c) {
    return di(l, n, u, c), Gf(l);
  }
  function Bn(l, n) {
    return di(l, null, null, n), Gf(l);
  }
  function yo(l, n, u) {
    l.lanes |= u;
    var c = l.alternate;
    c !== null && (c.lanes |= u);
    for (var r = !1, s = l.return; s !== null; )
      s.childLanes |= u, c = s.alternate, c !== null && (c.childLanes |= u), s.tag === 22 && (l = s.stateNode, l === null || l._visibility & 1 || (r = !0)), l = s, s = s.return;
    return l.tag === 3 ? (s = l.stateNode, r && n !== null && (r = 31 - Dl(u), l = s.hiddenUpdates, c = l[r], c === null ? l[r] = [n] : c.push(n), n.lane = u | 536870912), s) : null;
  }
  function Gf(l) {
    if (50 < Lo)
      throw Lo = 0, rm = null, Error(_(185));
    for (var n = l.return; n !== null; )
      l = n, n = l.return;
    return l.tag === 3 ? l.stateNode : null;
  }
  var mo = {};
  function kp(l, n, u, c) {
    this.tag = l, this.key = u, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.refCleanup = this.ref = null, this.pendingProps = n, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = c, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function oa(l, n, u, c) {
    return new kp(l, n, u, c);
  }
  function Lf(l) {
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
  function We(l, n) {
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
          return l = oa(31, u, n, r), l.elementType = de, l.lanes = s, l;
        case Ye:
          return Va(u.children, r, s, n);
        case Mt:
          y = 8, r |= 24;
          break;
        case $e:
          return l = oa(12, u, n, r | 2), l.elementType = $e, l.lanes = s, l;
        case De:
          return l = oa(13, u, n, r), l.elementType = De, l.lanes = s, l;
        case bt:
          return l = oa(19, u, n, r), l.elementType = bt, l.lanes = s, l;
        default:
          if (typeof l == "object" && l !== null)
            switch (l.$$typeof) {
              case tt:
              case Me:
                y = 10;
                break e;
              case Pt:
                y = 9;
                break e;
              case lt:
                y = 11;
                break e;
              case Ge:
                y = 14;
                break e;
              case Tt:
                y = 16, c = null;
                break e;
            }
          y = 29, u = Error(
            _(130, l === null ? "null" : typeof l, "")
          ), c = null;
      }
    return n = oa(y, u, n, r), n.elementType = l, n.type = c, n.lanes = s, n;
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
  var hi = [], yi = 0, Vf = null, vo = 0, Xa = [], fa = 0, zu = null, mn = 1, Qt = "";
  function ot(l, n) {
    hi[yi++] = vo, hi[yi++] = Vf, Vf = l, vo = n;
  }
  function Cs(l, n, u) {
    Xa[fa++] = mn, Xa[fa++] = Qt, Xa[fa++] = zu, zu = l;
    var c = mn;
    l = Qt;
    var r = 32 - Dl(c) - 1;
    c &= ~(1 << r), u += 1;
    var s = 32 - Dl(n) + r;
    if (30 < s) {
      var y = r - r % 5;
      s = (c & (1 << y) - 1).toString(32), c >>= y, r -= y, mn = 1 << 32 - Dl(n) + r | u << r | c, Qt = s + l;
    } else
      mn = 1 << s | u << r | c, Qt = l;
  }
  function rc(l) {
    l.return !== null && (ot(l, 1), Cs(l, 1, 0));
  }
  function Yn(l) {
    for (; l === Vf; )
      Vf = hi[--yi], hi[yi] = null, vo = hi[--yi], hi[yi] = null;
    for (; l === zu; )
      zu = Xa[--fa], Xa[fa] = null, Qt = Xa[--fa], Xa[fa] = null, mn = Xa[--fa], Xa[fa] = null;
  }
  var el = null, dt = null, st = !1, Qa = null, Za = !1, sc = Error(_(519));
  function Mu(l) {
    var n = Error(_(418, ""));
    throw So(Oa(n, l)), sc;
  }
  function Xf(l) {
    var n = l.stateNode, u = l.type, c = l.memoizedProps;
    switch (n[vl] = l, n[kl] = c, u) {
      case "dialog":
        Xe("cancel", n), Xe("close", n);
        break;
      case "iframe":
      case "object":
      case "embed":
        Xe("load", n);
        break;
      case "video":
      case "audio":
        for (u = 0; u < Mr.length; u++)
          Xe(Mr[u], n);
        break;
      case "source":
        Xe("error", n);
        break;
      case "img":
      case "image":
      case "link":
        Xe("error", n), Xe("load", n);
        break;
      case "details":
        Xe("toggle", n);
        break;
      case "input":
        Xe("invalid", n), ms(
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
        Xe("invalid", n);
        break;
      case "textarea":
        Xe("invalid", n), jh(n, c.value, c.defaultValue, c.children), ai(n);
    }
    u = c.children, typeof u != "string" && typeof u != "number" && typeof u != "bigint" || n.textContent === "" + u || c.suppressHydrationWarning === !0 || Rm(n.textContent, u) ? (c.popover != null && (Xe("beforetoggle", n), Xe("toggle", n)), c.onScroll != null && Xe("scroll", n), c.onScrollEnd != null && Xe("scrollend", n), c.onClick != null && (n.onclick = Ld), n = !0) : n = !1, n || Mu(l);
  }
  function sy(l) {
    for (el = l.return; el; )
      switch (el.tag) {
        case 5:
        case 13:
          Za = !1;
          return;
        case 27:
        case 3:
          Za = !0;
          return;
        default:
          el = el.return;
      }
  }
  function go(l) {
    if (l !== el) return !1;
    if (!st) return sy(l), st = !0, !1;
    var n = l.tag, u;
    if ((u = n !== 3 && n !== 27) && ((u = n === 5) && (u = l.type, u = !(u !== "form" && u !== "button") || nu(l.type, l.memoizedProps)), u = !u), u && dt && Mu(l), sy(l), n === 13) {
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
      n === 27 ? (n = dt, xi(l.type) ? (l = Hi, Hi = null, dt = l) : dt = n) : dt = el ? Tn(l.stateNode.nextSibling) : null;
    return !0;
  }
  function bo() {
    dt = el = null, st = !1;
  }
  function dy() {
    var l = Qa;
    return l !== null && (ma === null ? ma = l : ma.push.apply(
      ma,
      l
    ), Qa = null), l;
  }
  function So(l) {
    Qa === null ? Qa = [l] : Qa.push(l);
  }
  var Qf = q(null), Uu = null, pn = null;
  function _u(l, n, u) {
    I(Qf, n._currentValue), n._currentValue = u;
  }
  function jn(l) {
    l._currentValue = Qf.current, K(Qf);
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
  function To(l, n, u, c) {
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
          _l(r.pendingProps.value, y.value) || (l !== null ? l.push(p) : l = [p]);
        }
      } else if (r === il.current) {
        if (y = r.alternate, y === null) throw Error(_(387));
        y.memoizedState.memoizedState !== r.memoizedState.memoizedState && (l !== null ? l.push(ba) : l = [ba]);
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
      if (!_l(
        l.context._currentValue,
        l.memoizedValue
      ))
        return !0;
      l = l.next;
    }
    return !1;
  }
  function mi(l) {
    Uu = l, pn = null, l = l.dependencies, l !== null && (l.firstContext = null);
  }
  function gl(l) {
    return yy(Uu, l);
  }
  function Kf(l, n) {
    return Uu === null && mi(l), yy(l, n);
  }
  function yy(l, n) {
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
  }, Hs = M.unstable_scheduleCallback, $p = M.unstable_NormalPriority, fl = {
    $$typeof: Me,
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
  function Gn(l) {
    l.refCount--, l.refCount === 0 && Hs($p, function() {
      l.controller.abort();
    });
  }
  var pi = null, Jf = 0, Ka = 0, rl = null;
  function Ns(l, n) {
    if (pi === null) {
      var u = pi = [];
      Jf = 0, Ka = Hc(), rl = {
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
    if (--Jf === 0 && pi !== null) {
      rl !== null && (rl.status = "fulfilled");
      var l = pi;
      pi = null, Ka = 0, rl = null;
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
  var qs = O.S;
  O.S = function(l, n) {
    typeof n == "object" && n !== null && typeof n.then == "function" && Ns(l, n), qs !== null && qs(l, n);
  };
  var Ln = q(null);
  function kf() {
    var l = Ln.current;
    return l !== null ? l : Ut.pooledCache;
  }
  function dc(l, n) {
    n === null ? I(Ln, Ln.current) : I(Ln, n.pool);
  }
  function Bs() {
    var l = kf();
    return l === null ? null : { parent: fl._currentValue, pool: l };
  }
  var vi = Error(_(460)), Ys = Error(_(474)), $f = Error(_(542)), js = { then: function() {
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
        throw hc = n, vi;
    }
  }
  var hc = null;
  function py() {
    if (hc === null) throw Error(_(459));
    var l = hc;
    return hc = null, l;
  }
  function vy(l) {
    if (l === vi || l === $f)
      throw Error(_(483));
  }
  var Vn = !1;
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
  function ra(l) {
    return { lane: l, tag: 0, payload: null, callback: null, next: null };
  }
  function Xn(l, n, u) {
    var c = l.updateQueue;
    if (c === null) return null;
    if (c = c.shared, (gt & 2) !== 0) {
      var r = c.pending;
      return r === null ? n.next = n : (n.next = r.next, r.next = n), c.pending = n, n = Gf(l), yo(l, null, u), n;
    }
    return di(l, c, n, u), Gf(l);
  }
  function yc(l, n, u) {
    if (n = n.updateQueue, n !== null && (n = n.shared, (u & 4194048) !== 0)) {
      var c = n.lanes;
      c &= l.pendingLanes, u |= c, n.lanes = u, je(l, u);
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
  function Ro() {
    if (by) {
      var l = rl;
      if (l !== null) throw l;
    }
  }
  function Cu(l, n, u, c) {
    by = !1;
    var r = l.updateQueue;
    Vn = !1;
    var s = r.firstBaseUpdate, y = r.lastBaseUpdate, p = r.shared.pending;
    if (p !== null) {
      r.shared.pending = null;
      var S = p, x = S.next;
      S.next = null, y === null ? s = x : y.next = x, y = S;
      var Z = l.alternate;
      Z !== null && (Z = Z.updateQueue, p = Z.lastBaseUpdate, p !== y && (p === null ? Z.firstBaseUpdate = x : p.next = x, Z.lastBaseUpdate = S));
    }
    if (s !== null) {
      var $ = r.baseState;
      y = 0, Z = x = S = null, p = s;
      do {
        var w = p.lane & -536870913, Y = w !== p.lane;
        if (Y ? (nt & w) === w : (c & w) === w) {
          w !== 0 && w === Ka && (by = !0), Z !== null && (Z = Z.next = {
            lane: 0,
            tag: p.tag,
            payload: p.payload,
            callback: null,
            next: null
          });
          e: {
            var Ae = l, Re = p;
            w = n;
            var yt = u;
            switch (Re.tag) {
              case 1:
                if (Ae = Re.payload, typeof Ae == "function") {
                  $ = Ae.call(yt, $, w);
                  break e;
                }
                $ = Ae;
                break e;
              case 3:
                Ae.flags = Ae.flags & -65537 | 128;
              case 0:
                if (Ae = Re.payload, w = typeof Ae == "function" ? Ae.call(yt, $, w) : Ae, w == null) break e;
                $ = le({}, $, w);
                break e;
              case 2:
                Vn = !0;
            }
          }
          w = p.callback, w !== null && (l.flags |= 64, Y && (l.flags |= 8192), Y = r.callbacks, Y === null ? r.callbacks = [w] : Y.push(w));
        } else
          Y = {
            lane: w,
            tag: p.tag,
            payload: p.payload,
            callback: p.callback,
            next: null
          }, Z === null ? (x = Z = Y, S = $) : Z = Z.next = Y, y |= w;
        if (p = p.next, p === null) {
          if (p = r.shared.pending, p === null)
            break;
          Y = p, p = Y.next, Y.next = null, r.lastBaseUpdate = Y, r.shared.pending = null;
        }
      } while (!0);
      Z === null && (S = $), r.baseState = S, r.firstBaseUpdate = x, r.lastBaseUpdate = Z, s === null && (r.shared.lanes = 0), Yu |= y, l.lanes = y, l.memoizedState = $;
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
  var mc = q(null), If = q(0);
  function bl(l, n) {
    l = Bu, I(If, l), I(mc, n), Bu = l | n.baseLanes;
  }
  function Oo() {
    I(If, Bu), I(mc, mc.current);
  }
  function Do() {
    Bu = If.current, K(mc), K(If);
  }
  var Ja = 0, Ve = null, vt = null, Vt = null, Pf = !1, Da = !1, gi = !1, vn = 0, za = 0, xu = null, Sy = 0;
  function Xt() {
    throw Error(_(321));
  }
  function Qs(l, n) {
    if (n === null) return !1;
    for (var u = 0; u < n.length && u < l.length; u++)
      if (!_l(l[u], n[u])) return !1;
    return !0;
  }
  function Zs(l, n, u, c, r, s) {
    return Ja = s, Ve = n, n.memoizedState = null, n.updateQueue = null, n.lanes = 0, O.H = l === null || l.memoizedState === null ? wy : qy, gi = !1, s = u(c, r), gi = !1, Da && (s = Ty(
      n,
      u,
      c,
      r
    )), bi(l), s;
  }
  function bi(l) {
    O.H = od;
    var n = vt !== null && vt.next !== null;
    if (Ja = 0, Vt = vt = Ve = null, Pf = !1, za = 0, xu = null, n) throw Error(_(300));
    l === null || sl || (l = l.dependencies, l !== null && Zf(l) && (sl = !0));
  }
  function Ty(l, n, u, c) {
    Ve = l;
    var r = 0;
    do {
      if (Da && (xu = null), za = 0, Da = !1, 25 <= r) throw Error(_(301));
      if (r += 1, Vt = vt = null, l.updateQueue != null) {
        var s = l.updateQueue;
        s.lastEffect = null, s.events = null, s.stores = null, s.memoCache != null && (s.memoCache.index = 0);
      }
      O.H = Hu, s = n(u, c);
    } while (Da);
    return s;
  }
  function Fp() {
    var l = O.H, n = l.useState()[0];
    return n = typeof n.then == "function" ? tr(n) : n, l = l.useState()[0], (vt !== null ? vt.memoizedState : null) !== l && (Ve.flags |= 1024), n;
  }
  function Ks() {
    var l = vn !== 0;
    return vn = 0, l;
  }
  function zo(l, n, u) {
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
    Ja = 0, Vt = vt = Ve = null, Da = !1, za = vn = 0, xu = null;
  }
  function Ll() {
    var l = {
      memoizedState: null,
      baseState: null,
      baseQueue: null,
      queue: null,
      next: null
    };
    return Vt === null ? Ve.memoizedState = Vt = l : Vt = Vt.next = l, Vt;
  }
  function Zt() {
    if (vt === null) {
      var l = Ve.alternate;
      l = l !== null ? l.memoizedState : null;
    } else l = vt.next;
    var n = Vt === null ? Ve.memoizedState : Vt.next;
    if (n !== null)
      Vt = n, vt = l;
    else {
      if (l === null)
        throw Ve.alternate === null ? Error(_(467)) : Error(_(310));
      vt = l, l = {
        memoizedState: vt.memoizedState,
        baseState: vt.baseState,
        baseQueue: vt.baseQueue,
        queue: vt.queue,
        next: null
      }, Vt === null ? Ve.memoizedState = Vt = l : Vt = Vt.next = l;
    }
    return Vt;
  }
  function er() {
    return { lastEffect: null, events: null, stores: null, memoCache: null };
  }
  function tr(l) {
    var n = za;
    return za += 1, xu === null && (xu = []), l = my(xu, l, n), n = Ve, (Vt === null ? n.memoizedState : Vt.next) === null && (n = n.alternate, O.H = n === null || n.memoizedState === null ? wy : qy), l;
  }
  function al(l) {
    if (l !== null && typeof l == "object") {
      if (typeof l.then == "function") return tr(l);
      if (l.$$typeof === Me) return gl(l);
    }
    throw Error(_(438, String(l)));
  }
  function ks(l) {
    var n = null, u = Ve.updateQueue;
    if (u !== null && (n = u.memoCache), n == null) {
      var c = Ve.alternate;
      c !== null && (c = c.updateQueue, c !== null && (c = c.memoCache, c != null && (n = {
        data: c.data.map(function(r) {
          return r.slice();
        }),
        index: 0
      })));
    }
    if (n == null && (n = { data: [], index: 0 }), u === null && (u = er(), Ve.updateQueue = u), u.memoCache = n, u = n.data[n.index], u === void 0)
      for (u = n.data[n.index] = Array(l), c = 0; c < l; c++)
        u[c] = Rt;
    return n.index++, u;
  }
  function Qn(l, n) {
    return typeof n == "function" ? n(l) : n;
  }
  function lr(l) {
    var n = Zt();
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
      var p = y = null, S = null, x = n, Z = !1;
      do {
        var $ = x.lane & -536870913;
        if ($ !== x.lane ? (nt & $) === $ : (Ja & $) === $) {
          var w = x.revertLane;
          if (w === 0)
            S !== null && (S = S.next = {
              lane: 0,
              revertLane: 0,
              action: x.action,
              hasEagerState: x.hasEagerState,
              eagerState: x.eagerState,
              next: null
            }), $ === Ka && (Z = !0);
          else if ((Ja & w) === w) {
            x = x.next, w === Ka && (Z = !0);
            continue;
          } else
            $ = {
              lane: 0,
              revertLane: x.revertLane,
              action: x.action,
              hasEagerState: x.hasEagerState,
              eagerState: x.eagerState,
              next: null
            }, S === null ? (p = S = $, y = s) : S = S.next = $, Ve.lanes |= w, Yu |= w;
          $ = x.action, gi && u(s, $), s = x.hasEagerState ? x.eagerState : u(s, $);
        } else
          w = {
            lane: $,
            revertLane: x.revertLane,
            action: x.action,
            hasEagerState: x.hasEagerState,
            eagerState: x.eagerState,
            next: null
          }, S === null ? (p = S = w, y = s) : S = S.next = w, Ve.lanes |= $, Yu |= $;
        x = x.next;
      } while (x !== null && x !== n);
      if (S === null ? y = s : S.next = p, !_l(s, l.memoizedState) && (sl = !0, Z && (u = rl, u !== null)))
        throw u;
      l.memoizedState = s, l.baseState = y, l.baseQueue = S, c.lastRenderedState = s;
    }
    return r === null && (c.lanes = 0), [l.memoizedState, c.dispatch];
  }
  function Ws(l) {
    var n = Zt(), u = n.queue;
    if (u === null) throw Error(_(311));
    u.lastRenderedReducer = l;
    var c = u.dispatch, r = u.pending, s = n.memoizedState;
    if (r !== null) {
      u.pending = null;
      var y = r = r.next;
      do
        s = l(s, y.action), y = y.next;
      while (y !== r);
      _l(s, n.memoizedState) || (sl = !0), n.memoizedState = s, n.baseQueue === null && (n.baseState = s), u.lastRenderedState = s;
    }
    return [s, c];
  }
  function ar(l, n, u) {
    var c = Ve, r = Zt(), s = st;
    if (s) {
      if (u === void 0) throw Error(_(407));
      u = u();
    } else u = n();
    var y = !_l(
      (vt || r).memoizedState,
      u
    );
    y && (r.memoizedState = u, sl = !0), r = r.queue;
    var p = Ay.bind(null, c, r, l);
    if (Ot(2048, 8, p, [l]), r.getSnapshot !== n || y || Vt !== null && Vt.memoizedState.tag & 1) {
      if (c.flags |= 2048, sa(
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
      s || (Ja & 124) !== 0 || Fs(c, n, u);
    }
    return u;
  }
  function Fs(l, n, u) {
    l.flags |= 16384, l = { getSnapshot: n, value: u }, n = Ve.updateQueue, n === null ? (n = er(), Ve.updateQueue = n, n.stores = [l]) : (u = n.stores, u === null ? n.stores = [l] : u.push(l));
  }
  function Ey(l, n, u, c) {
    n.value = u, n.getSnapshot = c, Ry(n) && Is(l);
  }
  function Ay(l, n, u) {
    return u(function() {
      Ry(n) && Is(l);
    });
  }
  function Ry(l) {
    var n = l.getSnapshot;
    l = l.value;
    try {
      var u = n();
      return !_l(l, u);
    } catch {
      return !0;
    }
  }
  function Is(l) {
    var n = Bn(l, 2);
    n !== null && _a(n, l, 2);
  }
  function nr(l) {
    var n = Ll();
    if (typeof l == "function") {
      var u = l;
      if (l = u(), gi) {
        Ya(!0);
        try {
          u();
        } finally {
          Ya(!1);
        }
      }
    }
    return n.memoizedState = n.baseState = l, n.queue = {
      pending: null,
      lanes: 0,
      dispatch: null,
      lastRenderedReducer: Qn,
      lastRenderedState: l
    }, n;
  }
  function Ps(l, n, u, c) {
    return l.baseState = u, $s(
      l,
      vt,
      typeof c == "function" ? c : Qn
    );
  }
  function Ip(l, n, u, c, r) {
    if (bc(l)) throw Error(_(485));
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
      O.T !== null ? u(!0) : s.isTransition = !1, c(s), u = n.pending, u === null ? (s.next = n.pending = s, ed(n, s)) : (s.next = u.next, n.pending = u.next = s);
    }
  }
  function ed(l, n) {
    var u = n.action, c = n.payload, r = l.state;
    if (n.isTransition) {
      var s = O.T, y = {};
      O.T = y;
      try {
        var p = u(r, c), S = O.S;
        S !== null && S(y, p), ur(l, n, p);
      } catch (x) {
        ld(l, n, x);
      } finally {
        O.T = s;
      }
    } else
      try {
        s = u(r, c), ur(l, n, s);
      } catch (x) {
        ld(l, n, x);
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
    if (st) {
      var u = Ut.formState;
      if (u !== null) {
        e: {
          var c = Ve;
          if (st) {
            if (dt) {
              t: {
                for (var r = dt, s = Za; r.nodeType !== 8; ) {
                  if (!s) {
                    r = null;
                    break t;
                  }
                  if (r = Tn(
                    r.nextSibling
                  ), r === null) {
                    r = null;
                    break t;
                  }
                }
                s = r.data, r = s === "F!" || s === "F" ? r : null;
              }
              if (r) {
                dt = Tn(
                  r.nextSibling
                ), c = r.data === "F!";
                break e;
              }
            }
            Mu(c);
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
      lastRenderedReducer: ad,
      lastRenderedState: n
    }, u.queue = c, u = Hy.bind(
      null,
      Ve,
      c
    ), c.dispatch = u, c = nr(!1), s = fr.bind(
      null,
      Ve,
      !1,
      c.queue
    ), c = Ll(), r = {
      state: n,
      dispatch: null,
      action: l,
      pending: null
    }, c.queue = r, u = Ip.bind(
      null,
      Ve,
      r,
      s,
      u
    ), r.dispatch = u, c.memoizedState = l, [n, u, !1];
  }
  function Zn(l) {
    var n = Zt();
    return nd(n, vt, l);
  }
  function nd(l, n, u) {
    if (n = $s(
      l,
      n,
      ad
    )[0], l = lr(Qn)[0], typeof n == "object" && n !== null && typeof n.then == "function")
      try {
        var c = tr(n);
      } catch (y) {
        throw y === vi ? $f : y;
      }
    else c = n;
    n = Zt();
    var r = n.queue, s = r.dispatch;
    return u !== n.memoizedState && (Ve.flags |= 2048, sa(
      9,
      ir(),
      Ug.bind(null, r, u),
      null
    )), [c, s, l];
  }
  function Ug(l, n) {
    l.action = n;
  }
  function ud(l) {
    var n = Zt(), u = vt;
    if (u !== null)
      return nd(n, u, l);
    Zt(), n = n.memoizedState, u = Zt();
    var c = u.queue.dispatch;
    return u.memoizedState = l, [n, c, !1];
  }
  function sa(l, n, u, c) {
    return l = { tag: l, create: u, deps: c, inst: n, next: null }, n = Ve.updateQueue, n === null && (n = er(), Ve.updateQueue = n), u = n.lastEffect, u === null ? n.lastEffect = l.next = l : (c = u.next, u.next = l, l.next = c, n.lastEffect = l), l;
  }
  function ir() {
    return { destroy: void 0, resource: void 0 };
  }
  function cr() {
    return Zt().memoizedState;
  }
  function Si(l, n, u, c) {
    var r = Ll();
    c = c === void 0 ? null : c, Ve.flags |= l, r.memoizedState = sa(
      1 | n,
      ir(),
      u,
      c
    );
  }
  function Ot(l, n, u, c) {
    var r = Zt();
    c = c === void 0 ? null : c;
    var s = r.memoizedState.inst;
    vt !== null && c !== null && Qs(c, vt.memoizedState.deps) ? r.memoizedState = sa(n, s, u, c) : (Ve.flags |= l, r.memoizedState = sa(
      1 | n,
      s,
      u,
      c
    ));
  }
  function Pp(l, n) {
    Si(8390656, 8, l, n);
  }
  function ev(l, n) {
    Ot(2048, 8, l, n);
  }
  function zy(l, n) {
    return Ot(4, 2, l, n);
  }
  function gn(l, n) {
    return Ot(4, 4, l, n);
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
    u = u != null ? u.concat([l]) : null, Ot(4, 4, My.bind(null, n, l), u);
  }
  function pc() {
  }
  function vc(l, n) {
    var u = Zt();
    n = n === void 0 ? null : n;
    var c = u.memoizedState;
    return n !== null && Qs(n, c[1]) ? c[0] : (u.memoizedState = [l, n], l);
  }
  function Uy(l, n) {
    var u = Zt();
    n = n === void 0 ? null : n;
    var c = u.memoizedState;
    if (n !== null && Qs(n, c[1]))
      return c[0];
    if (c = l(), gi) {
      Ya(!0);
      try {
        l();
      } finally {
        Ya(!1);
      }
    }
    return u.memoizedState = [c, n], c;
  }
  function or(l, n, u) {
    return u === void 0 || (Ja & 1073741824) !== 0 ? l.memoizedState = n : (l.memoizedState = u, l = sm(), Ve.lanes |= l, Yu |= l, u);
  }
  function _y(l, n, u, c) {
    return _l(u, n) ? u : mc.current !== null ? (l = or(l, u, c), _l(l, n) || (sl = !0), l) : (Ja & 42) === 0 ? (sl = !0, l.memoizedState = u) : (l = sm(), Ve.lanes |= l, Yu |= l, n);
  }
  function tv(l, n, u, c, r) {
    var s = W.p;
    W.p = s !== 0 && 8 > s ? s : 8;
    var y = O.T, p = {};
    O.T = p, fr(l, !1, n, u);
    try {
      var S = r(), x = O.S;
      if (x !== null && x(p, S), S !== null && typeof S == "object" && typeof S.then == "function") {
        var Z = Wp(
          S,
          c
        );
        gc(
          l,
          n,
          Z,
          Ua(l)
        );
      } else
        gc(
          l,
          n,
          c,
          Ua(l)
        );
    } catch ($) {
      gc(
        l,
        n,
        { then: function() {
        }, status: "rejected", reason: $ },
        Ua()
      );
    } finally {
      W.p = s, O.T = y;
    }
  }
  function _g() {
  }
  function cd(l, n, u, c) {
    if (l.tag !== 5) throw Error(_(476));
    var r = lv(l).queue;
    tv(
      l,
      r,
      n,
      P,
      u === null ? _g : function() {
        return Mo(l), u(c);
      }
    );
  }
  function lv(l) {
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
        lastRenderedReducer: Qn,
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
        lastRenderedReducer: Qn,
        lastRenderedState: u
      },
      next: null
    }, l.memoizedState = n, l = l.alternate, l !== null && (l.memoizedState = n), n;
  }
  function Mo(l) {
    var n = lv(l).next.queue;
    gc(l, n, {}, Ua());
  }
  function ka() {
    return gl(ba);
  }
  function Cy() {
    return Zt().memoizedState;
  }
  function av() {
    return Zt().memoizedState;
  }
  function nv(l) {
    for (var n = l.return; n !== null; ) {
      switch (n.tag) {
        case 24:
        case 3:
          var u = Ua();
          l = ra(u);
          var c = Xn(n, l, u);
          c !== null && (_a(c, n, u), yc(c, n, u)), n = { cache: Ao() }, l.payload = n;
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
    }, bc(l) ? uv(n, u) : (u = ho(l, n, u, c), u !== null && (_a(u, l, c), Ny(u, n, c)));
  }
  function Hy(l, n, u) {
    var c = Ua();
    gc(l, n, u, c);
  }
  function gc(l, n, u, c) {
    var r = {
      lane: c,
      revertLane: 0,
      action: u,
      hasEagerState: !1,
      eagerState: null,
      next: null
    };
    if (bc(l)) uv(n, r);
    else {
      var s = l.alternate;
      if (l.lanes === 0 && (s === null || s.lanes === 0) && (s = n.lastRenderedReducer, s !== null))
        try {
          var y = n.lastRenderedState, p = s(y, u);
          if (r.hasEagerState = !0, r.eagerState = p, _l(p, y))
            return di(l, n, r, 0), Ut === null && hn(), !1;
        } catch {
        } finally {
        }
      if (u = ho(l, n, r, c), u !== null)
        return _a(u, l, c), Ny(u, n, c), !0;
    }
    return !1;
  }
  function fr(l, n, u, c) {
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
      ), n !== null && _a(n, l, 2);
  }
  function bc(l) {
    var n = l.alternate;
    return l === Ve || n !== null && n === Ve;
  }
  function uv(l, n) {
    Da = Pf = !0;
    var u = l.pending;
    u === null ? n.next = n : (n.next = u.next, u.next = n), l.pending = n;
  }
  function Ny(l, n, u) {
    if ((u & 4194048) !== 0) {
      var c = n.lanes;
      c &= l.pendingLanes, u |= c, n.lanes = u, je(l, u);
    }
  }
  var od = {
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
  }, wy = {
    readContext: gl,
    use: al,
    useCallback: function(l, n) {
      return Ll().memoizedState = [
        l,
        n === void 0 ? null : n
      ], l;
    },
    useContext: gl,
    useEffect: Pp,
    useImperativeHandle: function(l, n, u) {
      u = u != null ? u.concat([l]) : null, Si(
        4194308,
        4,
        My.bind(null, n, l),
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
        Ya(!0);
        try {
          l();
        } finally {
          Ya(!1);
        }
      }
      return u.memoizedState = [c, n], c;
    },
    useReducer: function(l, n, u) {
      var c = Ll();
      if (u !== void 0) {
        var r = u(n);
        if (gi) {
          Ya(!0);
          try {
            u(n);
          } finally {
            Ya(!1);
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
        Ve,
        l
      ), [c.memoizedState, l];
    },
    useRef: function(l) {
      var n = Ll();
      return l = { current: l }, n.memoizedState = l;
    },
    useState: function(l) {
      l = nr(l);
      var n = l.queue, u = Hy.bind(null, Ve, n);
      return n.dispatch = u, [l.memoizedState, u];
    },
    useDebugValue: pc,
    useDeferredValue: function(l, n) {
      var u = Ll();
      return or(u, l, n);
    },
    useTransition: function() {
      var l = nr(!1);
      return l = tv.bind(
        null,
        Ve,
        l.queue,
        !0,
        !1
      ), Ll().memoizedState = l, [!1, l];
    },
    useSyncExternalStore: function(l, n, u) {
      var c = Ve, r = Ll();
      if (st) {
        if (u === void 0)
          throw Error(_(407));
        u = u();
      } else {
        if (u = n(), Ut === null)
          throw Error(_(349));
        (nt & 124) !== 0 || Fs(c, n, u);
      }
      r.memoizedState = u;
      var s = { value: u, getSnapshot: n };
      return r.queue = s, Pp(Ay.bind(null, c, s, l), [
        l
      ]), c.flags |= 2048, sa(
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
      var l = Ll(), n = Ut.identifierPrefix;
      if (st) {
        var u = Qt, c = mn;
        u = (c & ~(1 << 32 - Dl(c) - 1)).toString(32) + u, n = "«" + n + "R" + u, u = vn++, 0 < u && (n += "H" + u.toString(32)), n += "»";
      } else
        u = Sy++, n = "«" + n + "r" + u.toString(32) + "»";
      return l.memoizedState = n;
    },
    useHostTransitionStatus: ka,
    useFormState: Dy,
    useActionState: Dy,
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
      return n.queue = u, n = fr.bind(
        null,
        Ve,
        !0,
        u
      ), u.dispatch = n, [l, n];
    },
    useMemoCache: ks,
    useCacheRefresh: function() {
      return Ll().memoizedState = nv.bind(
        null,
        Ve
      );
    }
  }, qy = {
    readContext: gl,
    use: al,
    useCallback: vc,
    useContext: gl,
    useEffect: ev,
    useImperativeHandle: id,
    useInsertionEffect: zy,
    useLayoutEffect: gn,
    useMemo: Uy,
    useReducer: lr,
    useRef: cr,
    useState: function() {
      return lr(Qn);
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
      var l = lr(Qn)[0], n = Zt().memoizedState;
      return [
        typeof l == "boolean" ? l : tr(l),
        n
      ];
    },
    useSyncExternalStore: ar,
    useId: Cy,
    useHostTransitionStatus: ka,
    useFormState: Zn,
    useActionState: Zn,
    useOptimistic: function(l, n) {
      var u = Zt();
      return Ps(u, vt, l, n);
    },
    useMemoCache: ks,
    useCacheRefresh: av
  }, Hu = {
    readContext: gl,
    use: al,
    useCallback: vc,
    useContext: gl,
    useEffect: ev,
    useImperativeHandle: id,
    useInsertionEffect: zy,
    useLayoutEffect: gn,
    useMemo: Uy,
    useReducer: Ws,
    useRef: cr,
    useState: function() {
      return Ws(Qn);
    },
    useDebugValue: pc,
    useDeferredValue: function(l, n) {
      var u = Zt();
      return vt === null ? or(u, l, n) : _y(
        u,
        vt.memoizedState,
        l,
        n
      );
    },
    useTransition: function() {
      var l = Ws(Qn)[0], n = Zt().memoizedState;
      return [
        typeof l == "boolean" ? l : tr(l),
        n
      ];
    },
    useSyncExternalStore: ar,
    useId: Cy,
    useHostTransitionStatus: ka,
    useFormState: ud,
    useActionState: ud,
    useOptimistic: function(l, n) {
      var u = Zt();
      return vt !== null ? Ps(u, vt, l, n) : (u.baseState = l, [l, u.queue.dispatch]);
    },
    useMemoCache: ks,
    useCacheRefresh: av
  }, Sc = null, Uo = 0;
  function fd(l) {
    var n = Uo;
    return Uo += 1, Sc === null && (Sc = []), my(Sc, l, n);
  }
  function Tc(l, n) {
    n = n.props.ref, l.ref = n !== void 0 ? n : null;
  }
  function Vl(l, n) {
    throw n.$$typeof === k ? Error(_(525)) : (l = Object.prototype.toString.call(n), Error(
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
  function da(l) {
    function n(z, R) {
      if (l) {
        var C = z.deletions;
        C === null ? (z.deletions = [R], z.flags |= 16) : C.push(R);
      }
    }
    function u(z, R) {
      if (!l) return null;
      for (; R !== null; )
        n(z, R), R = R.sibling;
      return null;
    }
    function c(z) {
      for (var R = /* @__PURE__ */ new Map(); z !== null; )
        z.key !== null ? R.set(z.key, z) : R.set(z.index, z), z = z.sibling;
      return R;
    }
    function r(z, R) {
      return z = yn(z, R), z.index = 0, z.sibling = null, z;
    }
    function s(z, R, C) {
      return z.index = C, l ? (C = z.alternate, C !== null ? (C = C.index, C < R ? (z.flags |= 67108866, R) : C) : (z.flags |= 67108866, R)) : (z.flags |= 1048576, R);
    }
    function y(z) {
      return l && z.alternate === null && (z.flags |= 67108866), z;
    }
    function p(z, R, C, J) {
      return R === null || R.tag !== 6 ? (R = po(C, z.mode, J), R.return = z, R) : (R = r(R, C), R.return = z, R);
    }
    function S(z, R, C, J) {
      var se = C.type;
      return se === Ye ? Z(
        z,
        R,
        C.props.children,
        J,
        C.key
      ) : R !== null && (R.elementType === se || typeof se == "object" && se !== null && se.$$typeof === Tt && By(se) === R.type) ? (R = r(R, C.props), Tc(R, C), R.return = z, R) : (R = ee(
        C.type,
        C.key,
        C.props,
        null,
        z.mode,
        J
      ), Tc(R, C), R.return = z, R);
    }
    function x(z, R, C, J) {
      return R === null || R.tag !== 4 || R.stateNode.containerInfo !== C.containerInfo || R.stateNode.implementation !== C.implementation ? (R = Lt(C, z.mode, J), R.return = z, R) : (R = r(R, C.children || []), R.return = z, R);
    }
    function Z(z, R, C, J, se) {
      return R === null || R.tag !== 7 ? (R = Va(
        C,
        z.mode,
        J,
        se
      ), R.return = z, R) : (R = r(R, C), R.return = z, R);
    }
    function $(z, R, C) {
      if (typeof R == "string" && R !== "" || typeof R == "number" || typeof R == "bigint")
        return R = po(
          "" + R,
          z.mode,
          C
        ), R.return = z, R;
      if (typeof R == "object" && R !== null) {
        switch (R.$$typeof) {
          case U:
            return C = ee(
              R.type,
              R.key,
              R.props,
              null,
              z.mode,
              C
            ), Tc(C, R), C.return = z, C;
          case ae:
            return R = Lt(
              R,
              z.mode,
              C
            ), R.return = z, R;
          case Tt:
            var J = R._init;
            return R = J(R._payload), $(z, R, C);
        }
        if (pt(R) || Ce(R))
          return R = Va(
            R,
            z.mode,
            C,
            null
          ), R.return = z, R;
        if (typeof R.then == "function")
          return $(z, fd(R), C);
        if (R.$$typeof === Me)
          return $(
            z,
            Kf(z, R),
            C
          );
        Vl(z, R);
      }
      return null;
    }
    function w(z, R, C, J) {
      var se = R !== null ? R.key : null;
      if (typeof C == "string" && C !== "" || typeof C == "number" || typeof C == "bigint")
        return se !== null ? null : p(z, R, "" + C, J);
      if (typeof C == "object" && C !== null) {
        switch (C.$$typeof) {
          case U:
            return C.key === se ? S(z, R, C, J) : null;
          case ae:
            return C.key === se ? x(z, R, C, J) : null;
          case Tt:
            return se = C._init, C = se(C._payload), w(z, R, C, J);
        }
        if (pt(C) || Ce(C))
          return se !== null ? null : Z(z, R, C, J, null);
        if (typeof C.then == "function")
          return w(
            z,
            R,
            fd(C),
            J
          );
        if (C.$$typeof === Me)
          return w(
            z,
            R,
            Kf(z, C),
            J
          );
        Vl(z, C);
      }
      return null;
    }
    function Y(z, R, C, J, se) {
      if (typeof J == "string" && J !== "" || typeof J == "number" || typeof J == "bigint")
        return z = z.get(C) || null, p(R, z, "" + J, se);
      if (typeof J == "object" && J !== null) {
        switch (J.$$typeof) {
          case U:
            return z = z.get(
              J.key === null ? C : J.key
            ) || null, S(R, z, J, se);
          case ae:
            return z = z.get(
              J.key === null ? C : J.key
            ) || null, x(R, z, J, se);
          case Tt:
            var Fe = J._init;
            return J = Fe(J._payload), Y(
              z,
              R,
              C,
              J,
              se
            );
        }
        if (pt(J) || Ce(J))
          return z = z.get(C) || null, Z(R, z, J, se, null);
        if (typeof J.then == "function")
          return Y(
            z,
            R,
            C,
            fd(J),
            se
          );
        if (J.$$typeof === Me)
          return Y(
            z,
            R,
            C,
            Kf(R, J),
            se
          );
        Vl(R, J);
      }
      return null;
    }
    function Ae(z, R, C, J) {
      for (var se = null, Fe = null, Ee = R, _e = R = 0, El = null; Ee !== null && _e < C.length; _e++) {
        Ee.index > _e ? (El = Ee, Ee = null) : El = Ee.sibling;
        var rt = w(
          z,
          Ee,
          C[_e],
          J
        );
        if (rt === null) {
          Ee === null && (Ee = El);
          break;
        }
        l && Ee && rt.alternate === null && n(z, Ee), R = s(rt, R, _e), Fe === null ? se = rt : Fe.sibling = rt, Fe = rt, Ee = El;
      }
      if (_e === C.length)
        return u(z, Ee), st && ot(z, _e), se;
      if (Ee === null) {
        for (; _e < C.length; _e++)
          Ee = $(z, C[_e], J), Ee !== null && (R = s(
            Ee,
            R,
            _e
          ), Fe === null ? se = Ee : Fe.sibling = Ee, Fe = Ee);
        return st && ot(z, _e), se;
      }
      for (Ee = c(Ee); _e < C.length; _e++)
        El = Y(
          Ee,
          z,
          _e,
          C[_e],
          J
        ), El !== null && (l && El.alternate !== null && Ee.delete(
          El.key === null ? _e : El.key
        ), R = s(
          El,
          R,
          _e
        ), Fe === null ? se = El : Fe.sibling = El, Fe = El);
      return l && Ee.forEach(function(Yi) {
        return n(z, Yi);
      }), st && ot(z, _e), se;
    }
    function Re(z, R, C, J) {
      if (C == null) throw Error(_(151));
      for (var se = null, Fe = null, Ee = R, _e = R = 0, El = null, rt = C.next(); Ee !== null && !rt.done; _e++, rt = C.next()) {
        Ee.index > _e ? (El = Ee, Ee = null) : El = Ee.sibling;
        var Yi = w(z, Ee, rt.value, J);
        if (Yi === null) {
          Ee === null && (Ee = El);
          break;
        }
        l && Ee && Yi.alternate === null && n(z, Ee), R = s(Yi, R, _e), Fe === null ? se = Yi : Fe.sibling = Yi, Fe = Yi, Ee = El;
      }
      if (rt.done)
        return u(z, Ee), st && ot(z, _e), se;
      if (Ee === null) {
        for (; !rt.done; _e++, rt = C.next())
          rt = $(z, rt.value, J), rt !== null && (R = s(rt, R, _e), Fe === null ? se = rt : Fe.sibling = rt, Fe = rt);
        return st && ot(z, _e), se;
      }
      for (Ee = c(Ee); !rt.done; _e++, rt = C.next())
        rt = Y(Ee, z, _e, rt.value, J), rt !== null && (l && rt.alternate !== null && Ee.delete(rt.key === null ? _e : rt.key), R = s(rt, R, _e), Fe === null ? se = rt : Fe.sibling = rt, Fe = rt);
      return l && Ee.forEach(function(Vg) {
        return n(z, Vg);
      }), st && ot(z, _e), se;
    }
    function yt(z, R, C, J) {
      if (typeof C == "object" && C !== null && C.type === Ye && C.key === null && (C = C.props.children), typeof C == "object" && C !== null) {
        switch (C.$$typeof) {
          case U:
            e: {
              for (var se = C.key; R !== null; ) {
                if (R.key === se) {
                  if (se = C.type, se === Ye) {
                    if (R.tag === 7) {
                      u(
                        z,
                        R.sibling
                      ), J = r(
                        R,
                        C.props.children
                      ), J.return = z, z = J;
                      break e;
                    }
                  } else if (R.elementType === se || typeof se == "object" && se !== null && se.$$typeof === Tt && By(se) === R.type) {
                    u(
                      z,
                      R.sibling
                    ), J = r(R, C.props), Tc(J, C), J.return = z, z = J;
                    break e;
                  }
                  u(z, R);
                  break;
                } else n(z, R);
                R = R.sibling;
              }
              C.type === Ye ? (J = Va(
                C.props.children,
                z.mode,
                J,
                C.key
              ), J.return = z, z = J) : (J = ee(
                C.type,
                C.key,
                C.props,
                null,
                z.mode,
                J
              ), Tc(J, C), J.return = z, z = J);
            }
            return y(z);
          case ae:
            e: {
              for (se = C.key; R !== null; ) {
                if (R.key === se)
                  if (R.tag === 4 && R.stateNode.containerInfo === C.containerInfo && R.stateNode.implementation === C.implementation) {
                    u(
                      z,
                      R.sibling
                    ), J = r(R, C.children || []), J.return = z, z = J;
                    break e;
                  } else {
                    u(z, R);
                    break;
                  }
                else n(z, R);
                R = R.sibling;
              }
              J = Lt(C, z.mode, J), J.return = z, z = J;
            }
            return y(z);
          case Tt:
            return se = C._init, C = se(C._payload), yt(
              z,
              R,
              C,
              J
            );
        }
        if (pt(C))
          return Ae(
            z,
            R,
            C,
            J
          );
        if (Ce(C)) {
          if (se = Ce(C), typeof se != "function") throw Error(_(150));
          return C = se.call(C), Re(
            z,
            R,
            C,
            J
          );
        }
        if (typeof C.then == "function")
          return yt(
            z,
            R,
            fd(C),
            J
          );
        if (C.$$typeof === Me)
          return yt(
            z,
            R,
            Kf(z, C),
            J
          );
        Vl(z, C);
      }
      return typeof C == "string" && C !== "" || typeof C == "number" || typeof C == "bigint" ? (C = "" + C, R !== null && R.tag === 6 ? (u(z, R.sibling), J = r(R, C), J.return = z, z = J) : (u(z, R), J = po(C, z.mode, J), J.return = z, z = J), y(z)) : u(z, R);
    }
    return function(z, R, C, J) {
      try {
        Uo = 0;
        var se = yt(
          z,
          R,
          C,
          J
        );
        return Sc = null, se;
      } catch (Ee) {
        if (Ee === vi || Ee === $f) throw Ee;
        var Fe = oa(29, Ee, null, z.mode);
        return Fe.lanes = J, Fe.return = z, Fe;
      } finally {
      }
    };
  }
  var Ec = da(!0), Kn = da(!1), Ma = q(null), Xl = null;
  function Nu(l) {
    var n = l.alternate;
    I(Dt, Dt.current & 1), I(Ma, l), Xl === null && (n === null || mc.current !== null || n.memoizedState !== null) && (Xl = l);
  }
  function Jn(l) {
    if (l.tag === 22) {
      if (I(Dt, Dt.current), I(Ma, l), Xl === null) {
        var n = l.alternate;
        n !== null && n.memoizedState !== null && (Xl = l);
      }
    } else kn();
  }
  function kn() {
    I(Dt, Dt.current), I(Ma, Ma.current);
  }
  function bn(l) {
    K(Ma), Xl === l && (Xl = null), K(Dt);
  }
  var Dt = q(0);
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
  function Ti(l, n, u, c) {
    n = l.memoizedState, u = u(c, n), u = u == null ? n : le({}, n, u), l.memoizedState = u, l.lanes === 0 && (l.updateQueue.baseState = u);
  }
  var rd = {
    enqueueSetState: function(l, n, u) {
      l = l._reactInternals;
      var c = Ua(), r = ra(c);
      r.payload = n, u != null && (r.callback = u), n = Xn(l, r, c), n !== null && (_a(n, l, c), yc(n, l, c));
    },
    enqueueReplaceState: function(l, n, u) {
      l = l._reactInternals;
      var c = Ua(), r = ra(c);
      r.tag = 1, r.payload = n, u != null && (r.callback = u), n = Xn(l, r, c), n !== null && (_a(n, l, c), yc(n, l, c));
    },
    enqueueForceUpdate: function(l, n) {
      l = l._reactInternals;
      var u = Ua(), c = ra(u);
      c.tag = 2, n != null && (c.callback = n), n = Xn(l, c, u), n !== null && (_a(n, l, u), yc(n, l, u));
    }
  };
  function _o(l, n, u, c, r, s, y) {
    return l = l.stateNode, typeof l.shouldComponentUpdate == "function" ? l.shouldComponentUpdate(c, s, y) : n.prototype && n.prototype.isPureReactComponent ? !oi(u, c) || !oi(r, s) : !0;
  }
  function Ac(l, n, u, c) {
    l = n.state, typeof n.componentWillReceiveProps == "function" && n.componentWillReceiveProps(u, c), typeof n.UNSAFE_componentWillReceiveProps == "function" && n.UNSAFE_componentWillReceiveProps(u, c), n.state !== l && rd.enqueueReplaceState(n, n.state, null);
  }
  function Ei(l, n) {
    var u = n;
    if ("ref" in n) {
      u = {};
      for (var c in n)
        c !== "ref" && (u[c] = n[c]);
    }
    if (l = l.defaultProps) {
      u === n && (u = le({}, u));
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
    } else if (typeof It == "object" && typeof It.emit == "function") {
      It.emit("uncaughtException", l);
      return;
    }
    console.error(l);
  };
  function Co(l) {
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
    return u = ra(u), u.tag = 3, u.payload = { element: null }, u.callback = function() {
      hr(l, n);
    }, u;
  }
  function Ly(l) {
    return l = ra(l), l.tag = 3, l;
  }
  function ha(l, n, u, c) {
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
      jy(n, u, c), typeof r != "function" && (Di === null ? Di = /* @__PURE__ */ new Set([this]) : Di.add(this));
      var p = c.stack;
      this.componentDidCatch(c.value, {
        componentStack: p !== null ? p : ""
      });
    });
  }
  function iv(l, n, u, c, r) {
    if (u.flags |= 32768, c !== null && typeof c == "object" && typeof c.then == "function") {
      if (n = u.alternate, n !== null && To(
        n,
        u,
        r,
        !0
      ), u = Ma.current, u !== null) {
        switch (u.tag) {
          case 13:
            return Xl === null ? xc() : u.alternate === null && $t === 0 && ($t = 3), u.flags &= -257, u.flags |= 65536, u.lanes = r, c === js ? u.flags |= 16384 : (n = u.updateQueue, n === null ? u.updateQueue = /* @__PURE__ */ new Set([c]) : n.add(c), qd(l, c, r)), !1;
          case 22:
            return u.flags |= 65536, c === js ? u.flags |= 16384 : (n = u.updateQueue, n === null ? (n = {
              transitions: null,
              markerInstances: null,
              retryQueue: /* @__PURE__ */ new Set([c])
            }, u.updateQueue = n) : (u = n.retryQueue, u === null ? n.retryQueue = /* @__PURE__ */ new Set([c]) : u.add(c)), qd(l, c, r)), !1;
        }
        throw Error(_(435, u.tag));
      }
      return qd(l, c, r), xc(), !1;
    }
    if (st)
      return n = Ma.current, n !== null ? ((n.flags & 65536) === 0 && (n.flags |= 256), n.flags |= 65536, n.lanes = r, c !== sc && (l = Error(_(422), { cause: c }), So(Oa(l, u)))) : (c !== sc && (n = Error(_(423), {
        cause: c
      }), So(
        Oa(n, u)
      )), l = l.current.alternate, l.flags |= 65536, r &= -r, l.lanes |= r, c = Oa(c, u), r = Gy(
        l.stateNode,
        c,
        r
      ), gy(l, r), $t !== 4 && ($t = 2)), !1;
    var s = Error(_(520), { cause: c });
    if (s = Oa(s, u), Yo === null ? Yo = [s] : Yo.push(s), $t !== 4 && ($t = 2), n === null) return !0;
    c = Oa(c, u), u = n;
    do {
      switch (u.tag) {
        case 3:
          return u.flags |= 65536, l = r & -r, u.lanes |= l, l = Gy(u.stateNode, c, l), gy(u, l), !1;
        case 1:
          if (n = u.type, s = u.stateNode, (u.flags & 128) === 0 && (typeof n.getDerivedStateFromError == "function" || s !== null && typeof s.componentDidCatch == "function" && (Di === null || !Di.has(s))))
            return u.flags |= 65536, r &= -r, u.lanes |= r, r = Ly(r), ha(
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
  var Kt = Error(_(461)), sl = !1;
  function Sl(l, n, u, c) {
    n.child = l === null ? Kn(n, null, u, c) : Ec(
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
    return mi(n), c = Zs(
      l,
      n,
      u,
      y,
      s,
      r
    ), p = Ks(), l !== null && !sl ? (zo(l, n, r), $n(l, n, r)) : (st && p && rc(n), n.flags |= 1, Sl(l, n, c, r), n.child);
  }
  function wu(l, n, u, c, r) {
    if (l === null) {
      var s = u.type;
      return typeof s == "function" && !Lf(s) && s.defaultProps === void 0 && u.compare === null ? (n.tag = 15, n.type = s, Rc(
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
      if (u = u.compare, u = u !== null ? u : oi, u(y, c) && l.ref === n.ref)
        return $n(l, n, r);
    }
    return n.flags |= 1, l = yn(s, c), l.ref = n.ref, l.return = n, n.child = l;
  }
  function Rc(l, n, u, c, r) {
    if (l !== null) {
      var s = l.memoizedProps;
      if (oi(s, c) && l.ref === n.ref)
        if (sl = !1, n.pendingProps = c = s, Sd(l, r))
          (l.flags & 131072) !== 0 && (sl = !0);
        else
          return n.lanes = l.lanes, $n(l, n, r);
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
          s !== null ? s.cachePool : null
        ), s !== null ? bl(n, s) : Oo(), Jn(n);
      else
        return n.lanes = n.childLanes = 536870912, Oc(
          l,
          n,
          s !== null ? s.baseLanes | u : u,
          u
        );
    } else
      s !== null ? (dc(n, s.cachePool), bl(n, s), kn(), n.memoizedState = null) : (l !== null && dc(n, null), Oo(), kn());
    return Sl(l, n, r, u), n.child;
  }
  function Oc(l, n, u, c) {
    var r = kf();
    return r = r === null ? null : { parent: fl._currentValue, pool: r }, n.memoizedState = {
      baseLanes: u,
      cachePool: r
    }, l !== null && dc(n, null), Oo(), Jn(n), l !== null && To(l, n, c, !0), null;
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
    return mi(n), u = Zs(
      l,
      n,
      u,
      c,
      void 0,
      r
    ), c = Ks(), l !== null && !sl ? (zo(l, n, r), $n(l, n, r)) : (st && c && rc(n), n.flags |= 1, Sl(l, n, u, r), n.child);
  }
  function Vy(l, n, u, c, r, s) {
    return mi(n), n.updateQueue = null, u = Ty(
      n,
      c,
      u,
      r
    ), bi(l), c = Ks(), l !== null && !sl ? (zo(l, n, s), $n(l, n, s)) : (st && c && rc(n), n.flags |= 1, Sl(l, n, u, s), n.child);
  }
  function hd(l, n, u, c, r) {
    if (mi(n), n.stateNode === null) {
      var s = mo, y = u.contextType;
      typeof y == "object" && y !== null && (s = gl(y)), s = new u(c, s), n.memoizedState = s.state !== null && s.state !== void 0 ? s.state : null, s.updater = rd, n.stateNode = s, s._reactInternals = n, s = n.stateNode, s.props = c, s.state = n.memoizedState, s.refs = {}, Ls(n), y = u.contextType, s.context = typeof y == "object" && y !== null ? gl(y) : mo, s.state = n.memoizedState, y = u.getDerivedStateFromProps, typeof y == "function" && (Ti(
        n,
        u,
        y,
        c
      ), s.state = n.memoizedState), typeof u.getDerivedStateFromProps == "function" || typeof s.getSnapshotBeforeUpdate == "function" || typeof s.UNSAFE_componentWillMount != "function" && typeof s.componentWillMount != "function" || (y = s.state, typeof s.componentWillMount == "function" && s.componentWillMount(), typeof s.UNSAFE_componentWillMount == "function" && s.UNSAFE_componentWillMount(), y !== s.state && rd.enqueueReplaceState(s, s.state, null), Cu(n, c, s, r), Ro(), s.state = n.memoizedState), typeof s.componentDidMount == "function" && (n.flags |= 4194308), c = !0;
    } else if (l === null) {
      s = n.stateNode;
      var p = n.memoizedProps, S = Ei(u, p);
      s.props = S;
      var x = s.context, Z = u.contextType;
      y = mo, typeof Z == "object" && Z !== null && (y = gl(Z));
      var $ = u.getDerivedStateFromProps;
      Z = typeof $ == "function" || typeof s.getSnapshotBeforeUpdate == "function", p = n.pendingProps !== p, Z || typeof s.UNSAFE_componentWillReceiveProps != "function" && typeof s.componentWillReceiveProps != "function" || (p || x !== y) && Ac(
        n,
        s,
        c,
        y
      ), Vn = !1;
      var w = n.memoizedState;
      s.state = w, Cu(n, c, s, r), Ro(), x = n.memoizedState, p || w !== x || Vn ? (typeof $ == "function" && (Ti(
        n,
        u,
        $,
        c
      ), x = n.memoizedState), (S = Vn || _o(
        n,
        u,
        S,
        c,
        w,
        x,
        y
      )) ? (Z || typeof s.UNSAFE_componentWillMount != "function" && typeof s.componentWillMount != "function" || (typeof s.componentWillMount == "function" && s.componentWillMount(), typeof s.UNSAFE_componentWillMount == "function" && s.UNSAFE_componentWillMount()), typeof s.componentDidMount == "function" && (n.flags |= 4194308)) : (typeof s.componentDidMount == "function" && (n.flags |= 4194308), n.memoizedProps = c, n.memoizedState = x), s.props = c, s.state = x, s.context = y, c = S) : (typeof s.componentDidMount == "function" && (n.flags |= 4194308), c = !1);
    } else {
      s = n.stateNode, Vs(l, n), y = n.memoizedProps, Z = Ei(u, y), s.props = Z, $ = n.pendingProps, w = s.context, x = u.contextType, S = mo, typeof x == "object" && x !== null && (S = gl(x)), p = u.getDerivedStateFromProps, (x = typeof p == "function" || typeof s.getSnapshotBeforeUpdate == "function") || typeof s.UNSAFE_componentWillReceiveProps != "function" && typeof s.componentWillReceiveProps != "function" || (y !== $ || w !== S) && Ac(
        n,
        s,
        c,
        S
      ), Vn = !1, w = n.memoizedState, s.state = w, Cu(n, c, s, r), Ro();
      var Y = n.memoizedState;
      y !== $ || w !== Y || Vn || l !== null && l.dependencies !== null && Zf(l.dependencies) ? (typeof p == "function" && (Ti(
        n,
        u,
        p,
        c
      ), Y = n.memoizedState), (Z = Vn || _o(
        n,
        u,
        Z,
        c,
        w,
        Y,
        S
      ) || l !== null && l.dependencies !== null && Zf(l.dependencies)) ? (x || typeof s.UNSAFE_componentWillUpdate != "function" && typeof s.componentWillUpdate != "function" || (typeof s.componentWillUpdate == "function" && s.componentWillUpdate(c, Y, S), typeof s.UNSAFE_componentWillUpdate == "function" && s.UNSAFE_componentWillUpdate(
        c,
        Y,
        S
      )), typeof s.componentDidUpdate == "function" && (n.flags |= 4), typeof s.getSnapshotBeforeUpdate == "function" && (n.flags |= 1024)) : (typeof s.componentDidUpdate != "function" || y === l.memoizedProps && w === l.memoizedState || (n.flags |= 4), typeof s.getSnapshotBeforeUpdate != "function" || y === l.memoizedProps && w === l.memoizedState || (n.flags |= 1024), n.memoizedProps = c, n.memoizedState = Y), s.props = c, s.state = Y, s.context = S, c = Z) : (typeof s.componentDidUpdate != "function" || y === l.memoizedProps && w === l.memoizedState || (n.flags |= 4), typeof s.getSnapshotBeforeUpdate != "function" || y === l.memoizedProps && w === l.memoizedState || (n.flags |= 1024), c = !1);
    }
    return s = c, yr(l, n), c = (n.flags & 128) !== 0, s || c ? (s = n.stateNode, u = c && typeof u.getDerivedStateFromError != "function" ? null : s.render(), n.flags |= 1, l !== null && c ? (n.child = Ec(
      n,
      l.child,
      null,
      r
    ), n.child = Ec(
      n,
      null,
      u,
      r
    )) : Sl(l, n, u, r), n.memoizedState = s.state, l = n.child) : l = $n(
      l,
      n,
      r
    ), l;
  }
  function yd(l, n, u, c) {
    return bo(), n.flags |= 256, Sl(l, n, u, c), n.child;
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
    return l = l !== null ? l.childLanes & ~u : 0, n && (l |= Fa), l;
  }
  function Zy(l, n, u) {
    var c = n.pendingProps, r = !1, s = (n.flags & 128) !== 0, y;
    if ((y = s) || (y = l !== null && l.memoizedState === null ? !1 : (Dt.current & 2) !== 0), y && (r = !0, n.flags &= -129), y = (n.flags & 32) !== 0, n.flags &= -33, l === null) {
      if (st) {
        if (r ? Nu(n) : kn(), st) {
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
              treeContext: zu !== null ? { id: mn, overflow: Qt } : null,
              retryLane: 536870912,
              hydrationErrors: null
            }, S = oa(
              18,
              null,
              null,
              0
            ), S.stateNode = p, S.return = n, n.child = S, el = n, dt = null, S = !0) : S = !1;
          }
          S || Mu(n);
        }
        if (p = n.memoizedState, p !== null && (p = p.dehydrated, p !== null))
          return Hr(p) ? n.lanes = 32 : n.lanes = 536870912, null;
        bn(n);
      }
      return p = c.children, c = c.fallback, r ? (kn(), r = n.mode, p = vd(
        { mode: "hidden", children: p },
        r
      ), c = Va(
        c,
        r,
        u,
        null
      ), p.return = n, c.return = n, p.sibling = c, n.child = p, r = n.child, r.memoizedState = Xy(u), r.childLanes = Qy(
        l,
        y,
        u
      ), n.memoizedState = md, c) : (Nu(n), pd(n, p));
    }
    if (S = l.memoizedState, S !== null && (p = S.dehydrated, p !== null)) {
      if (s)
        n.flags & 256 ? (Nu(n), n.flags &= -257, n = Ai(
          l,
          n,
          u
        )) : n.memoizedState !== null ? (kn(), n.child = l.child, n.flags |= 128, n = null) : (kn(), r = c.fallback, p = n.mode, c = vd(
          { mode: "visible", children: c.children },
          p
        ), r = Va(
          r,
          p,
          u,
          null
        ), r.flags |= 2, c.return = n, r.return = n, c.sibling = r, n.child = c, Ec(
          n,
          l.child,
          null,
          u
        ), c = n.child, c.memoizedState = Xy(u), c.childLanes = Qy(
          l,
          y,
          u
        ), n.memoizedState = md, n = r);
      else if (Nu(n), Hr(p)) {
        if (y = p.nextSibling && p.nextSibling.dataset, y) var x = y.dgst;
        y = x, c = Error(_(419)), c.stack = "", c.digest = y, So({ value: c, source: null, stack: null }), n = Ai(
          l,
          n,
          u
        );
      } else if (sl || To(l, n, u, !1), y = (u & l.childLanes) !== 0, sl || y) {
        if (y = Ut, y !== null && (c = u & -u, c = (c & 42) !== 0 ? 1 : ll(c), c = (c & (y.suspendedLanes | u)) !== 0 ? 0 : c, c !== 0 && c !== S.retryLane))
          throw S.retryLane = c, Bn(l, c), _a(y, l, c), Kt;
        p.data === "$?" || xc(), n = Ai(
          l,
          n,
          u
        );
      } else
        p.data === "$?" ? (n.flags |= 192, n.child = l.child, n = null) : (l = S.treeContext, dt = Tn(
          p.nextSibling
        ), el = n, st = !0, Qa = null, Za = !1, l !== null && (Xa[fa++] = mn, Xa[fa++] = Qt, Xa[fa++] = zu, mn = l.id, Qt = l.overflow, zu = n), n = pd(
          n,
          c.children
        ), n.flags |= 4096);
      return n;
    }
    return r ? (kn(), r = c.fallback, p = n.mode, S = l.child, x = S.sibling, c = yn(S, {
      mode: "hidden",
      children: c.children
    }), c.subtreeFlags = S.subtreeFlags & 65011712, x !== null ? r = yn(x, r) : (r = Va(
      r,
      p,
      u,
      null
    ), r.flags |= 2), r.return = n, c.return = n, c.sibling = r, n.child = c, c = r, r = n.child, p = l.child.memoizedState, p === null ? p = Xy(u) : (S = p.cachePool, S !== null ? (x = fl._currentValue, S = S.parent !== x ? { parent: x, pool: x } : S) : S = Bs(), p = {
      baseLanes: p.baseLanes | u,
      cachePool: S
    }), r.memoizedState = p, r.childLanes = Qy(
      l,
      y,
      u
    ), n.memoizedState = md, c) : (Nu(n), u = l.child, l = u.sibling, u = yn(u, {
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
    return l = oa(22, l, null, n), l.lanes = 0, l.stateNode = {
      _visibility: 1,
      _pendingMarkers: null,
      _retryCache: null,
      _transitions: null
    }, l;
  }
  function Ai(l, n, u) {
    return Ec(n, l.child, null, u), l = pd(
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
    if (Sl(l, n, c.children, u), c = Dt.current, (c & 2) !== 0)
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
    switch (I(Dt, c), r) {
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
  function $n(l, n, u) {
    if (l !== null && (n.dependencies = l.dependencies), Yu |= n.lanes, (u & n.childLanes) === 0)
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
  function Sd(l, n) {
    return (l.lanes & n) !== 0 ? !0 : (l = l.dependencies, !!(l !== null && Zf(l)));
  }
  function ov(l, n, u) {
    switch (n.tag) {
      case 3:
        Ne(n, n.stateNode.containerInfo), _u(n, fl, l.memoizedState.cache), bo();
        break;
      case 27:
      case 5:
        na(n);
        break;
      case 4:
        Ne(n, n.stateNode.containerInfo);
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
          return c.dehydrated !== null ? (Nu(n), n.flags |= 128, null) : (u & n.child.childLanes) !== 0 ? Zy(l, n, u) : (Nu(n), l = $n(
            l,
            n,
            u
          ), l !== null ? l.sibling : null);
        Nu(n);
        break;
      case 19:
        var r = (l.flags & 128) !== 0;
        if (c = (u & n.childLanes) !== 0, c || (To(
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
        if (r = n.memoizedState, r !== null && (r.rendering = null, r.tail = null, r.lastEffect = null), I(Dt, Dt.current), c) break;
        return null;
      case 22:
      case 23:
        return n.lanes = 0, sd(l, n, u);
      case 24:
        _u(n, fl, l.memoizedState.cache);
    }
    return $n(l, n, u);
  }
  function fv(l, n, u) {
    if (l !== null)
      if (l.memoizedProps !== n.pendingProps)
        sl = !0;
      else {
        if (!Sd(l, u) && (n.flags & 128) === 0)
          return sl = !1, ov(
            l,
            n,
            u
          );
        sl = (l.flags & 131072) !== 0;
      }
    else
      sl = !1, st && (n.flags & 1048576) !== 0 && Cs(n, vo, n.index);
    switch (n.lanes = 0, n.tag) {
      case 16:
        e: {
          l = n.pendingProps;
          var c = n.elementType, r = c._init;
          if (c = r(c._payload), n.type = c, typeof c == "function")
            Lf(c) ? (l = Ei(c, l), n.tag = 1, n = hd(
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
              if (r = c.$$typeof, r === lt) {
                n.tag = 11, n = cv(
                  null,
                  n,
                  c,
                  l,
                  u
                );
                break e;
              } else if (r === Ge) {
                n.tag = 14, n = wu(
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
        return dd(
          l,
          n,
          n.type,
          n.pendingProps,
          u
        );
      case 1:
        return c = n.type, r = Ei(
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
          if (Ne(
            n,
            n.stateNode.containerInfo
          ), l === null) throw Error(_(387));
          c = n.pendingProps;
          var s = n.memoizedState;
          r = s.element, Vs(l, n), Cu(n, c, null, u);
          var y = n.memoizedState;
          if (c = y.cache, _u(n, fl, c), c !== s.cache && hy(
            n,
            [fl],
            u,
            !0
          ), Ro(), c = y.element, s.isDehydrated)
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
              r = Oa(
                Error(_(424)),
                n
              ), So(r), n = yd(
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
              for (dt = Tn(l.firstChild), el = n, st = !0, Qa = null, Za = !0, u = Kn(
                n,
                null,
                c,
                u
              ), n.child = u; u; )
                u.flags = u.flags & -3 | 4096, u = u.sibling;
            }
          else {
            if (bo(), c === r) {
              n = $n(
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
        return yr(l, n), l === null ? (u = Ov(
          n.type,
          null,
          n.pendingProps,
          null
        )) ? n.memoizedState = u : st || (u = n.type, l = n.pendingProps, c = Pa(
          oe.current
        ).createElement(u), c[vl] = n, c[kl] = l, xe(c, u, l), ol(c), n.stateNode = c) : n.memoizedState = Ov(
          n.type,
          l.memoizedProps,
          n.pendingProps,
          l.memoizedState
        ), null;
      case 27:
        return na(n), l === null && st && (c = n.stateNode = fe(
          n.type,
          n.pendingProps,
          oe.current
        ), el = n, Za = !0, r = dt, xi(n.type) ? (Hi = r, dt = Tn(
          c.firstChild
        )) : dt = r), Sl(
          l,
          n,
          n.pendingProps.children,
          u
        ), yr(l, n), l === null && (n.flags |= 4194304), n.child;
      case 5:
        return l === null && st && ((r = c = dt) && (c = Wo(
          c,
          n.type,
          n.pendingProps,
          Za
        ), c !== null ? (n.stateNode = c, el = n, dt = Tn(
          c.firstChild
        ), Za = !1, r = !0) : r = !1), r || Mu(n)), na(n), r = n.type, s = n.pendingProps, y = l !== null ? l.memoizedProps : null, c = s.children, nu(r, s) ? c = null : y !== null && nu(r, y) && (n.flags |= 32), n.memoizedState !== null && (r = Zs(
          l,
          n,
          Fp,
          null,
          null,
          u
        ), ba._currentValue = r), yr(l, n), Sl(l, n, c, u), n.child;
      case 6:
        return l === null && st && ((l = u = dt) && (u = jg(
          u,
          n.pendingProps,
          Za
        ), u !== null ? (n.stateNode = u, el = n, dt = null, l = !0) : l = !1), l || Mu(n)), null;
      case 13:
        return Zy(l, n, u);
      case 4:
        return Ne(
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
        return cv(
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
        return r = n.type._context, c = n.pendingProps.children, mi(n), r = gl(r), c = c(r), n.flags |= 1, Sl(l, n, c, u), n.child;
      case 14:
        return wu(
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
        return bd(l, n, u);
      case 31:
        return c = n.pendingProps, u = n.mode, c = {
          mode: c.mode,
          children: c.children
        }, l === null ? (u = vd(
          c,
          u
        ), u.ref = n.ref, n.child = u, u.return = n, n = u) : (u = yn(l.child, c), u.ref = n.ref, n.child = u, u.return = n, n = u), n;
      case 22:
        return sd(l, n, u);
      case 24:
        return mi(n), c = gl(fl), l === null ? (r = kf(), r === null && (r = Ut, s = Ao(), r.pooledCache = s, s.refCount++, s !== null && (r.pooledCacheLanes |= u), r = s), n.memoizedState = {
          parent: c,
          cache: r
        }, Ls(n), _u(n, fl, r)) : ((l.lanes & u) !== 0 && (Vs(l, n), Cu(n, null, null, u), Ro()), r = l.memoizedState, s = n.memoizedState, r.parent !== c ? (r = { parent: c, cache: c }, n.memoizedState = r, n.lanes === 0 && (n.memoizedState = n.updateQueue.baseState = r), _u(n, fl, c)) : (c = s.cache, _u(n, fl, c), c !== r.cache && hy(
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
  function Wn(l) {
    l.flags |= 4;
  }
  function xo(l, n) {
    if (n.type !== "stylesheet" || (n.state.loading & 4) !== 0)
      l.flags &= -16777217;
    else if (l.flags |= 16777216, !Mm(n)) {
      if (n = Ma.current, n !== null && ((nt & 4194048) === nt ? Xl !== null : (nt & 62914560) !== nt && (nt & 536870912) === 0 || n !== Xl))
        throw hc = js, Ys;
      l.flags |= 8192;
    }
  }
  function pr(l, n) {
    n !== null && (l.flags |= 4), l.flags & 16384 && (n = l.tag !== 22 ? ue() : 536870912, l.lanes |= n, Bo |= n);
  }
  function Ho(l, n) {
    if (!st)
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
  function Ue(l) {
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
    switch (Yn(n), n.tag) {
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
        return Ue(n), null;
      case 1:
        return Ue(n), null;
      case 3:
        return u = n.stateNode, c = null, l !== null && (c = l.memoizedState.cache), n.memoizedState.cache !== c && (n.flags |= 2048), jn(fl), wt(), u.pendingContext && (u.context = u.pendingContext, u.pendingContext = null), (l === null || l.child === null) && (go(n) ? Wn(n) : l === null || l.memoizedState.isDehydrated && (n.flags & 256) === 0 || (n.flags |= 1024, dy())), Ue(n), null;
      case 26:
        return u = n.memoizedState, l === null ? (Wn(n), u !== null ? (Ue(n), xo(n, u)) : (Ue(n), n.flags &= -16777217)) : u ? u !== l.memoizedState ? (Wn(n), Ue(n), xo(n, u)) : (Ue(n), n.flags &= -16777217) : (l.memoizedProps !== c && Wn(n), Ue(n), n.flags &= -16777217), null;
      case 27:
        zn(n), u = oe.current;
        var r = n.type;
        if (l !== null && n.stateNode != null)
          l.memoizedProps !== c && Wn(n);
        else {
          if (!c) {
            if (n.stateNode === null)
              throw Error(_(166));
            return Ue(n), null;
          }
          l = ce.current, go(n) ? Xf(n) : (l = fe(r, c, u), n.stateNode = l, Wn(n));
        }
        return Ue(n), null;
      case 5:
        if (zn(n), u = n.type, l !== null && n.stateNode != null)
          l.memoizedProps !== c && Wn(n);
        else {
          if (!c) {
            if (n.stateNode === null)
              throw Error(_(166));
            return Ue(n), null;
          }
          if (l = ce.current, go(n))
            Xf(n);
          else {
            switch (r = Pa(
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
            l[vl] = n, l[kl] = c;
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
            e: switch (xe(l, u, c), u) {
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
            l && Wn(n);
          }
        }
        return Ue(n), n.flags &= -16777217, null;
      case 6:
        if (l && n.stateNode != null)
          l.memoizedProps !== c && Wn(n);
        else {
          if (typeof c != "string" && n.stateNode === null)
            throw Error(_(166));
          if (l = oe.current, go(n)) {
            if (l = n.stateNode, u = n.memoizedProps, c = null, r = el, r !== null)
              switch (r.tag) {
                case 27:
                case 5:
                  c = r.memoizedProps;
              }
            l[vl] = n, l = !!(l.nodeValue === u || c !== null && c.suppressHydrationWarning === !0 || Rm(l.nodeValue, u)), l || Mu(n);
          } else
            l = Pa(l).createTextNode(
              c
            ), l[vl] = n, n.stateNode = l;
        }
        return Ue(n), null;
      case 13:
        if (c = n.memoizedState, l === null || l.memoizedState !== null && l.memoizedState.dehydrated !== null) {
          if (r = go(n), c !== null && c.dehydrated !== null) {
            if (l === null) {
              if (!r) throw Error(_(318));
              if (r = n.memoizedState, r = r !== null ? r.dehydrated : null, !r) throw Error(_(317));
              r[vl] = n;
            } else
              bo(), (n.flags & 128) === 0 && (n.memoizedState = null), n.flags |= 4;
            Ue(n), r = !1;
          } else
            r = dy(), l !== null && l.memoizedState !== null && (l.memoizedState.hydrationErrors = r), r = !0;
          if (!r)
            return n.flags & 256 ? (bn(n), n) : (bn(n), null);
        }
        if (bn(n), (n.flags & 128) !== 0)
          return n.lanes = u, n;
        if (u = c !== null, l = l !== null && l.memoizedState !== null, u) {
          c = n.child, r = null, c.alternate !== null && c.alternate.memoizedState !== null && c.alternate.memoizedState.cachePool !== null && (r = c.alternate.memoizedState.cachePool.pool);
          var s = null;
          c.memoizedState !== null && c.memoizedState.cachePool !== null && (s = c.memoizedState.cachePool.pool), s !== r && (c.flags |= 2048);
        }
        return u !== l && u && (n.child.flags |= 8192), pr(n, n.updateQueue), Ue(n), null;
      case 4:
        return wt(), l === null && Em(n.stateNode.containerInfo), Ue(n), null;
      case 10:
        return jn(n.type), Ue(n), null;
      case 19:
        if (K(Dt), r = n.memoizedState, r === null) return Ue(n), null;
        if (c = (n.flags & 128) !== 0, s = r.rendering, s === null)
          if (c) Ho(r, !1);
          else {
            if ($t !== 0 || l !== null && (l.flags & 128) !== 0)
              for (l = n.child; l !== null; ) {
                if (s = rr(l), s !== null) {
                  for (n.flags |= 128, Ho(r, !1), l = s.updateQueue, n.updateQueue = l, pr(n, l), n.subtreeFlags = 0, l = u, u = n.child; u !== null; )
                    We(u, l), u = u.sibling;
                  return I(
                    Dt,
                    Dt.current & 1 | 2
                  ), n.child;
                }
                l = l.sibling;
              }
            r.tail !== null && pl() > Ud && (n.flags |= 128, c = !0, Ho(r, !1), n.lanes = 4194304);
          }
        else {
          if (!c)
            if (l = rr(s), l !== null) {
              if (n.flags |= 128, c = !0, l = l.updateQueue, n.updateQueue = l, pr(n, l), Ho(r, !0), r.tail === null && r.tailMode === "hidden" && !s.alternate && !st)
                return Ue(n), null;
            } else
              2 * pl() - r.renderingStartTime > Ud && u !== 536870912 && (n.flags |= 128, c = !0, Ho(r, !1), n.lanes = 4194304);
          r.isBackwards ? (s.sibling = n.child, n.child = s) : (l = r.last, l !== null ? l.sibling = s : n.child = s, r.last = s);
        }
        return r.tail !== null ? (n = r.tail, r.rendering = n, r.tail = n.sibling, r.renderingStartTime = pl(), n.sibling = null, l = Dt.current, I(Dt, c ? l & 1 | 2 : l & 1), n) : (Ue(n), null);
      case 22:
      case 23:
        return bn(n), Do(), c = n.memoizedState !== null, l !== null ? l.memoizedState !== null !== c && (n.flags |= 8192) : c && (n.flags |= 8192), c ? (u & 536870912) !== 0 && (n.flags & 128) === 0 && (Ue(n), n.subtreeFlags & 6 && (n.flags |= 8192)) : Ue(n), u = n.updateQueue, u !== null && pr(n, u.retryQueue), u = null, l !== null && l.memoizedState !== null && l.memoizedState.cachePool !== null && (u = l.memoizedState.cachePool.pool), c = null, n.memoizedState !== null && n.memoizedState.cachePool !== null && (c = n.memoizedState.cachePool.pool), c !== u && (n.flags |= 2048), l !== null && K(Ln), null;
      case 24:
        return u = null, l !== null && (u = l.memoizedState.cache), n.memoizedState.cache !== u && (n.flags |= 2048), jn(fl), Ue(n), null;
      case 25:
        return null;
      case 30:
        return null;
    }
    throw Error(_(156, n.tag));
  }
  function Cg(l, n) {
    switch (Yn(n), n.tag) {
      case 1:
        return l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 3:
        return jn(fl), wt(), l = n.flags, (l & 65536) !== 0 && (l & 128) === 0 ? (n.flags = l & -65537 | 128, n) : null;
      case 26:
      case 27:
      case 5:
        return zn(n), null;
      case 13:
        if (bn(n), l = n.memoizedState, l !== null && l.dehydrated !== null) {
          if (n.alternate === null)
            throw Error(_(340));
          bo();
        }
        return l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 19:
        return K(Dt), null;
      case 4:
        return wt(), null;
      case 10:
        return jn(n.type), null;
      case 22:
      case 23:
        return bn(n), Do(), l !== null && K(Ln), l = n.flags, l & 65536 ? (n.flags = l & -65537 | 128, n) : null;
      case 24:
        return jn(fl), null;
      case 25:
        return null;
      default:
        return null;
    }
  }
  function Jy(l, n) {
    switch (Yn(n), n.tag) {
      case 3:
        jn(fl), wt();
        break;
      case 26:
      case 27:
      case 5:
        zn(n);
        break;
      case 4:
        wt();
        break;
      case 13:
        bn(n);
        break;
      case 19:
        K(Dt);
        break;
      case 10:
        jn(n.type);
        break;
      case 22:
      case 23:
        bn(n), Do(), l !== null && K(Ln);
        break;
      case 24:
        jn(fl);
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
      Et(n, n.return, p);
    }
  }
  function Ri(l, n, u) {
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
              var S = u, x = p;
              try {
                x();
              } catch (Z) {
                Et(
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
      Et(n, n.return, Z);
    }
  }
  function Td(l) {
    var n = l.updateQueue;
    if (n !== null) {
      var u = l.stateNode;
      try {
        Ff(n, u);
      } catch (c) {
        Et(l, l.return, c);
      }
    }
  }
  function ky(l, n, u) {
    u.props = Ei(
      l.type,
      l.memoizedProps
    ), u.state = l.memoizedState;
    try {
      u.componentWillUnmount();
    } catch (c) {
      Et(l, n, c);
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
    } catch (r) {
      Et(l, n, r);
    }
  }
  function Sn(l, n) {
    var u = l.ref, c = l.refCleanup;
    if (u !== null)
      if (typeof c == "function")
        try {
          c();
        } catch (r) {
          Et(l, n, r);
        } finally {
          l.refCleanup = null, l = l.alternate, l != null && (l.refCleanup = null);
        }
      else if (typeof u == "function")
        try {
          u(null);
        } catch (r) {
          Et(l, n, r);
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
    } catch (r) {
      Et(l, l.return, r);
    }
  }
  function $y(l, n, u) {
    try {
      var c = l.stateNode;
      qg(c, l.type, u, n), c[kl] = n;
    } catch (r) {
      Et(l, l.return, r);
    }
  }
  function rv(l) {
    return l.tag === 5 || l.tag === 3 || l.tag === 26 || l.tag === 27 && xi(l.type) || l.tag === 4;
  }
  function $a(l) {
    e: for (; ; ) {
      for (; l.sibling === null; ) {
        if (l.return === null || rv(l.return)) return null;
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
      l = l.stateNode, n ? (u.nodeType === 9 ? u.body : u.nodeName === "HTML" ? u.ownerDocument.body : u).insertBefore(l, n) : (n = u.nodeType === 9 ? u.body : u.nodeName === "HTML" ? u.ownerDocument.body : u, n.appendChild(l), u = u._reactRootContainer, u != null || n.onclick !== null || (n.onclick = Ld));
    else if (c !== 4 && (c === 27 && xi(l.type) && (u = l.stateNode, n = null), l = l.child, l !== null))
      for (Dc(l, n, u), l = l.sibling; l !== null; )
        Dc(l, n, u), l = l.sibling;
  }
  function Ed(l, n, u) {
    var c = l.tag;
    if (c === 5 || c === 6)
      l = l.stateNode, n ? u.insertBefore(l, n) : u.appendChild(l);
    else if (c !== 4 && (c === 27 && xi(l.type) && (u = l.stateNode), l = l.child, l !== null))
      for (Ed(l, n, u), l = l.sibling; l !== null; )
        Ed(l, n, u), l = l.sibling;
  }
  function Ad(l) {
    var n = l.stateNode, u = l.memoizedProps;
    try {
      for (var c = l.type, r = n.attributes; r.length; )
        n.removeAttributeNode(r[0]);
      xe(n, c, u), n[vl] = l, n[kl] = u;
    } catch (s) {
      Et(l, l.return, s);
    }
  }
  var Fn = !1, Jt = !1, Rd = !1, Od = typeof WeakSet == "function" ? WeakSet : Set, dl = null;
  function Wy(l, n) {
    if (l = l.containerInfo, _r = qr, l = uy(l), jf(l)) {
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
            var y = 0, p = -1, S = -1, x = 0, Z = 0, $ = l, w = null;
            t: for (; ; ) {
              for (var Y; $ !== u || r !== 0 && $.nodeType !== 3 || (p = y + r), $ !== s || c !== 0 && $.nodeType !== 3 || (S = y + c), $.nodeType === 3 && (y += $.nodeValue.length), (Y = $.firstChild) !== null; )
                w = $, $ = Y;
              for (; ; ) {
                if ($ === l) break t;
                if (w === u && ++x === r && (p = y), w === s && ++Z === c && (S = y), (Y = $.nextSibling) !== null) break;
                $ = w, w = $.parentNode;
              }
              $ = Y;
            }
            u = p === -1 || S === -1 ? null : { start: p, end: S };
          } else u = null;
        }
      u = u || { start: 0, end: 0 };
    } else u = null;
    for (Cr = { focusedElem: l, selectionRange: u }, qr = !1, dl = n; dl !== null; )
      if (n = dl, l = n.child, (n.subtreeFlags & 1024) !== 0 && l !== null)
        l.return = n, dl = l;
      else
        for (; dl !== null; ) {
          switch (n = dl, s = n.alternate, l = n.flags, n.tag) {
            case 0:
              break;
            case 11:
            case 15:
              break;
            case 1:
              if ((l & 1024) !== 0 && s !== null) {
                l = void 0, u = n, r = s.memoizedProps, s = s.memoizedState, c = u.stateNode;
                try {
                  var Ae = Ei(
                    u.type,
                    r,
                    u.elementType === u.type
                  );
                  l = c.getSnapshotBeforeUpdate(
                    Ae,
                    s
                  ), c.__reactInternalSnapshotBeforeUpdate = l;
                } catch (Re) {
                  Et(
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
            l.return = n.return, dl = l;
            break;
          }
          dl = n.return;
        }
  }
  function Fy(l, n, u) {
    var c = u.flags;
    switch (u.tag) {
      case 0:
      case 11:
      case 15:
        Pn(l, u), c & 4 && vr(5, u);
        break;
      case 1:
        if (Pn(l, u), c & 4)
          if (l = u.stateNode, n === null)
            try {
              l.componentDidMount();
            } catch (y) {
              Et(u, u.return, y);
            }
          else {
            var r = Ei(
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
              Et(
                u,
                u.return,
                y
              );
            }
          }
        c & 64 && Td(u), c & 512 && No(u, u.return);
        break;
      case 3:
        if (Pn(l, u), c & 64 && (l = u.updateQueue, l !== null)) {
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
            Et(u, u.return, y);
          }
        }
        break;
      case 27:
        n === null && c & 4 && Ad(u);
      case 26:
      case 5:
        Pn(l, u), n === null && c & 4 && wo(u), c & 512 && No(u, u.return);
        break;
      case 12:
        Pn(l, u);
        break;
      case 13:
        Pn(l, u), c & 4 && Dd(l, u), c & 64 && (l = u.memoizedState, l !== null && (l = l.dehydrated, l !== null && (u = xg.bind(
          null,
          u
        ), Gg(l, u))));
        break;
      case 22:
        if (c = u.memoizedState !== null || Fn, !c) {
          n = n !== null && n.memoizedState !== null || Jt, r = Fn;
          var s = Jt;
          Fn = c, (Jt = n) && !s ? Oi(
            l,
            u,
            (u.subtreeFlags & 8772) !== 0
          ) : Pn(l, u), Fn = r, Jt = s;
        }
        break;
      case 30:
        break;
      default:
        Pn(l, u);
    }
  }
  function Iy(l) {
    var n = l.alternate;
    n !== null && (l.alternate = null, Iy(n)), l.child = null, l.deletions = null, l.sibling = null, l.tag === 5 && (n = l.stateNode, n !== null && Af(n)), l.stateNode = null, l.return = null, l.dependencies = null, l.memoizedProps = null, l.memoizedState = null, l.pendingProps = null, l.stateNode = null, l.updateQueue = null;
  }
  var qt = null, Cl = !1;
  function In(l, n, u) {
    for (u = u.child; u !== null; )
      et(l, n, u), u = u.sibling;
  }
  function et(l, n, u) {
    if (Ol && typeof Ol.onCommitFiberUnmount == "function")
      try {
        Ol.onCommitFiberUnmount(Pu, u);
      } catch {
      }
    switch (u.tag) {
      case 26:
        Jt || Sn(u, n), In(
          l,
          n,
          u
        ), u.memoizedState ? u.memoizedState.count-- : u.stateNode && (u = u.stateNode, u.parentNode.removeChild(u));
        break;
      case 27:
        Jt || Sn(u, n);
        var c = qt, r = Cl;
        xi(u.type) && (qt = u.stateNode, Cl = !1), In(
          l,
          n,
          u
        ), va(u.stateNode), qt = c, Cl = r;
        break;
      case 5:
        Jt || Sn(u, n);
      case 6:
        if (c = qt, r = Cl, qt = null, In(
          l,
          n,
          u
        ), qt = c, Cl = r, qt !== null)
          if (Cl)
            try {
              (qt.nodeType === 9 ? qt.body : qt.nodeName === "HTML" ? qt.ownerDocument.body : qt).removeChild(u.stateNode);
            } catch (s) {
              Et(
                u,
                n,
                s
              );
            }
          else
            try {
              qt.removeChild(u.stateNode);
            } catch (s) {
              Et(
                u,
                n,
                s
              );
            }
        break;
      case 18:
        qt !== null && (Cl ? (l = qt, Xd(
          l.nodeType === 9 ? l.body : l.nodeName === "HTML" ? l.ownerDocument.body : l,
          u.stateNode
        ), cu(l)) : Xd(qt, u.stateNode));
        break;
      case 4:
        c = qt, r = Cl, qt = u.stateNode.containerInfo, Cl = !0, In(
          l,
          n,
          u
        ), qt = c, Cl = r;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        Jt || Ri(2, u, n), Jt || Ri(4, u, n), In(
          l,
          n,
          u
        );
        break;
      case 1:
        Jt || (Sn(u, n), c = u.stateNode, typeof c.componentWillUnmount == "function" && ky(
          u,
          n,
          c
        )), In(
          l,
          n,
          u
        );
        break;
      case 21:
        In(
          l,
          n,
          u
        );
        break;
      case 22:
        Jt = (c = Jt) || u.memoizedState !== null, In(
          l,
          n,
          u
        ), Jt = c;
        break;
      default:
        In(
          l,
          n,
          u
        );
    }
  }
  function Dd(l, n) {
    if (n.memoizedState === null && (l = n.alternate, l !== null && (l = l.memoizedState, l !== null && (l = l.dehydrated, l !== null))))
      try {
        cu(l);
      } catch (u) {
        Et(n, n.return, u);
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
  function Fl(l, n) {
    var u = n.deletions;
    if (u !== null)
      for (var c = 0; c < u.length; c++) {
        var r = u[c], s = l, y = n, p = y;
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
        et(s, y, r), qt = null, Cl = !1, s = r.alternate, s !== null && (s.return = null), r.return = null;
      }
    if (n.subtreeFlags & 13878)
      for (n = n.child; n !== null; )
        gr(n, l), n = n.sibling;
  }
  var Il = null;
  function gr(l, n) {
    var u = l.alternate, c = l.flags;
    switch (l.tag) {
      case 0:
      case 11:
      case 14:
      case 15:
        Fl(n, l), Tl(l), c & 4 && (Ri(3, l, l.return), vr(3, l), Ri(5, l, l.return));
        break;
      case 1:
        Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), c & 64 && Fn && (l = l.updateQueue, l !== null && (c = l.callbacks, c !== null && (u = l.shared.hiddenCallbacks, l.shared.hiddenCallbacks = u === null ? c : u.concat(c))));
        break;
      case 26:
        var r = Il;
        if (Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), c & 4) {
          var s = u !== null ? u.memoizedState : null;
          if (c = l.memoizedState, u === null)
            if (c === null)
              if (l.stateNode === null) {
                e: {
                  c = l.type, u = l.memoizedProps, r = r.ownerDocument || r;
                  t: switch (c) {
                    case "title":
                      s = r.getElementsByTagName("title")[0], (!s || s[ye] || s[vl] || s.namespaceURI === "http://www.w3.org/2000/svg" || s.hasAttribute("itemprop")) && (s = r.createElement(c), r.head.insertBefore(
                        s,
                        r.querySelector("head > title")
                      )), xe(s, c, u), s[vl] = l, ol(s), c = s;
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
                      s = r.createElement(c), xe(s, c, u), r.head.appendChild(s);
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
                      s = r.createElement(c), xe(s, c, u), r.head.appendChild(s);
                      break;
                    default:
                      throw Error(_(468, c));
                  }
                  s[vl] = l, ol(s), c = s;
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
        Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), u !== null && c & 4 && $y(
          l,
          l.memoizedProps,
          u.memoizedProps
        );
        break;
      case 5:
        if (Fl(n, l), Tl(l), c & 512 && (Jt || u === null || Sn(u, u.return)), l.flags & 32) {
          r = l.stateNode;
          try {
            uo(r, "");
          } catch (Y) {
            Et(l, l.return, Y);
          }
        }
        c & 4 && l.stateNode != null && (r = l.memoizedProps, $y(
          l,
          r,
          u !== null ? u.memoizedProps : r
        )), c & 1024 && (Rd = !0);
        break;
      case 6:
        if (Fl(n, l), Tl(l), c & 4) {
          if (l.stateNode === null)
            throw Error(_(162));
          c = l.memoizedProps, u = l.stateNode;
          try {
            u.nodeValue = c;
          } catch (Y) {
            Et(l, l.return, Y);
          }
        }
        break;
      case 3:
        if (qi = null, r = Il, Il = Qd(n.containerInfo), Fl(n, l), Il = r, Tl(l), c & 4 && u !== null && u.memoizedState.isDehydrated)
          try {
            cu(n.containerInfo);
          } catch (Y) {
            Et(l, l.return, Y);
          }
        Rd && (Rd = !1, em(l));
        break;
      case 4:
        c = Il, Il = Qd(
          l.stateNode.containerInfo
        ), Fl(n, l), Tl(l), Il = c;
        break;
      case 12:
        Fl(n, l), Tl(l);
        break;
      case 13:
        Fl(n, l), Tl(l), l.child.flags & 8192 && l.memoizedState !== null != (u !== null && u.memoizedState !== null) && (om = pl()), c & 4 && (c = l.updateQueue, c !== null && (l.updateQueue = null, zd(l, c)));
        break;
      case 22:
        r = l.memoizedState !== null;
        var S = u !== null && u.memoizedState !== null, x = Fn, Z = Jt;
        if (Fn = x || r, Jt = Z || S, Fl(n, l), Jt = Z, Fn = x, Tl(l), c & 8192)
          e: for (n = l.stateNode, n._visibility = r ? n._visibility & -2 : n._visibility | 1, r && (u === null || S || Fn || Jt || Bt(l)), u = null, n = l; ; ) {
            if (n.tag === 5 || n.tag === 26) {
              if (u === null) {
                S = u = n;
                try {
                  if (s = S.stateNode, r)
                    y = s.style, typeof y.setProperty == "function" ? y.setProperty("display", "none", "important") : y.display = "none";
                  else {
                    p = S.stateNode;
                    var $ = S.memoizedProps.style, w = $ != null && $.hasOwnProperty("display") ? $.display : null;
                    p.style.display = w == null || typeof w == "boolean" ? "" : ("" + w).trim();
                  }
                } catch (Y) {
                  Et(S, S.return, Y);
                }
              }
            } else if (n.tag === 6) {
              if (u === null) {
                S = n;
                try {
                  S.stateNode.nodeValue = r ? "" : S.memoizedProps;
                } catch (Y) {
                  Et(S, S.return, Y);
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
        Fl(n, l), Tl(l), c & 4 && (c = l.updateQueue, c !== null && (l.updateQueue = null, zd(l, c)));
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
          if (rv(c)) {
            u = c;
            break;
          }
          c = c.return;
        }
        if (u == null) throw Error(_(160));
        switch (u.tag) {
          case 27:
            var r = u.stateNode, s = $a(l);
            Ed(l, s, r);
            break;
          case 5:
            var y = u.stateNode;
            u.flags & 32 && (uo(y, ""), u.flags &= -33);
            var p = $a(l);
            Ed(l, p, y);
            break;
          case 3:
          case 4:
            var S = u.stateNode.containerInfo, x = $a(l);
            Dc(
              l,
              x,
              S
            );
            break;
          default:
            throw Error(_(161));
        }
      } catch (Z) {
        Et(l, l.return, Z);
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
  function Pn(l, n) {
    if (n.subtreeFlags & 8772)
      for (n = n.child; n !== null; )
        Fy(l, n.alternate, n), n = n.sibling;
  }
  function Bt(l) {
    for (l = l.child; l !== null; ) {
      var n = l;
      switch (n.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          Ri(4, n, n.return), Bt(n);
          break;
        case 1:
          Sn(n, n.return);
          var u = n.stateNode;
          typeof u.componentWillUnmount == "function" && ky(
            n,
            n.return,
            u
          ), Bt(n);
          break;
        case 27:
          va(n.stateNode);
        case 26:
        case 5:
          Sn(n, n.return), Bt(n);
          break;
        case 22:
          n.memoizedState === null && Bt(n);
          break;
        case 30:
          Bt(n);
          break;
        default:
          Bt(n);
      }
      l = l.sibling;
    }
  }
  function Oi(l, n, u) {
    for (u = u && (n.subtreeFlags & 8772) !== 0, n = n.child; n !== null; ) {
      var c = n.alternate, r = l, s = n, y = s.flags;
      switch (s.tag) {
        case 0:
        case 11:
        case 15:
          Oi(
            r,
            s,
            u
          ), vr(4, s);
          break;
        case 1:
          if (Oi(
            r,
            s,
            u
          ), c = s, r = c.stateNode, typeof r.componentDidMount == "function")
            try {
              r.componentDidMount();
            } catch (x) {
              Et(c, c.return, x);
            }
          if (c = s, r = c.updateQueue, r !== null) {
            var p = c.stateNode;
            try {
              var S = r.shared.hiddenCallbacks;
              if (S !== null)
                for (r.shared.hiddenCallbacks = null, r = 0; r < S.length; r++)
                  Xs(S[r], p);
            } catch (x) {
              Et(c, c.return, x);
            }
          }
          u && y & 64 && Td(s), No(s, s.return);
          break;
        case 27:
          Ad(s);
        case 26:
        case 5:
          Oi(
            r,
            s,
            u
          ), u && c === null && y & 4 && wo(s), No(s, s.return);
          break;
        case 12:
          Oi(
            r,
            s,
            u
          );
          break;
        case 13:
          Oi(
            r,
            s,
            u
          ), u && y & 4 && Dd(r, s);
          break;
        case 22:
          s.memoizedState === null && Oi(
            r,
            s,
            u
          ), No(s, s.return);
          break;
        case 30:
          break;
        default:
          Oi(
            r,
            s,
            u
          );
      }
      n = n.sibling;
    }
  }
  function Wa(l, n) {
    var u = null;
    l !== null && l.memoizedState !== null && l.memoizedState.cachePool !== null && (u = l.memoizedState.cachePool.pool), l = null, n.memoizedState !== null && n.memoizedState.cachePool !== null && (l = n.memoizedState.cachePool.pool), l !== u && (l != null && l.refCount++, u != null && Gn(u));
  }
  function Md(l, n) {
    l = null, n.alternate !== null && (l = n.alternate.memoizedState.cache), n = n.memoizedState.cache, n !== l && (n.refCount++, l != null && Gn(l));
  }
  function xl(l, n, u, c) {
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
        xl(
          l,
          n,
          u,
          c
        ), r & 2048 && vr(9, n);
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
        ), r & 2048 && (l = null, n.alternate !== null && (l = n.alternate.memoizedState.cache), n = n.memoizedState.cache, n !== l && (n.refCount++, l != null && Gn(l)));
        break;
      case 12:
        if (r & 2048) {
          xl(
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
            Et(n, n.return, S);
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
        s = n.stateNode, y = n.alternate, n.memoizedState !== null ? s._visibility & 2 ? xl(
          l,
          n,
          u,
          c
        ) : ht(l, n) : s._visibility & 2 ? xl(
          l,
          n,
          u,
          c
        ) : (s._visibility |= 2, qu(
          l,
          n,
          u,
          c,
          (n.subtreeFlags & 10256) !== 0
        )), r & 2048 && Wa(y, n);
        break;
      case 24:
        xl(
          l,
          n,
          u,
          c
        ), r & 2048 && Md(n.alternate, n);
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
  function qu(l, n, u, c, r) {
    for (r = r && (n.subtreeFlags & 10256) !== 0, n = n.child; n !== null; ) {
      var s = l, y = n, p = u, S = c, x = y.flags;
      switch (y.tag) {
        case 0:
        case 11:
        case 15:
          qu(
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
          y.memoizedState !== null ? Z._visibility & 2 ? qu(
            s,
            y,
            p,
            S,
            r
          ) : ht(
            s,
            y
          ) : (Z._visibility |= 2, qu(
            s,
            y,
            p,
            S,
            r
          )), r && x & 2048 && Wa(
            y.alternate,
            y
          );
          break;
        case 24:
          qu(
            s,
            y,
            p,
            S,
            r
          ), r && x & 2048 && Md(y.alternate, y);
          break;
        default:
          qu(
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
            ht(u, c), r & 2048 && Wa(
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
        Il = Qd(l.stateNode.containerInfo), kt(l), Il = n;
        break;
      case 22:
        l.memoizedState === null && (n = l.alternate, n !== null && n.memoizedState !== null ? (n = zc, zc = 16777216, kt(l), zc = n) : kt(l));
        break;
      default:
        kt(l);
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
  function Mc(l) {
    var n = l.deletions;
    if ((l.flags & 16) !== 0) {
      if (n !== null)
        for (var u = 0; u < n.length; u++) {
          var c = n[u];
          dl = c, nm(
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
          dl = c, nm(
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
  function nm(l, n) {
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
          Gn(u.memoizedState.cache);
      }
      if (c = u.child, c !== null) c.return = u, dl = c;
      else
        e: for (u = l; dl !== null; ) {
          c = dl;
          var r = c.sibling, s = c.return;
          if (Iy(c), c === u) {
            dl = null;
            break e;
          }
          if (r !== null) {
            r.return = s, dl = r;
            break e;
          }
          dl = s;
        }
    }
  }
  var um = {
    getCacheForType: function(l) {
      var n = gl(fl), u = n.data.get(l);
      return u === void 0 && (u = l(), n.data.set(l, u)), u;
    }
  }, dv = typeof WeakMap == "function" ? WeakMap : Map, gt = 0, Ut = null, at = null, nt = 0, St = 0, ya = null, eu = !1, qo = !1, im = !1, Bu = 0, $t = 0, Yu = 0, Uc = 0, tu = 0, Fa = 0, Bo = 0, Yo = null, ma = null, cm = !1, om = 0, Ud = 1 / 0, jo = null, Di = null, Hl = 0, lu = null, Go = null, Nl = 0, _d = 0, Cd = null, fm = null, Lo = 0, rm = null;
  function Ua() {
    if ((gt & 2) !== 0 && nt !== 0)
      return nt & -nt;
    if (O.T !== null) {
      var l = Ka;
      return l !== 0 ? l : Hc();
    }
    return fs();
  }
  function sm() {
    Fa === 0 && (Fa = (nt & 536870912) === 0 || st ? te() : 536870912);
    var l = Ma.current;
    return l !== null && (l.flags |= 32), Fa;
  }
  function _a(l, n, u) {
    (l === Ut && (St === 2 || St === 9) || l.cancelPendingCommit !== null) && (au(l, 0), ju(
      l,
      nt,
      Fa,
      !1
    )), we(l, u), ((gt & 2) === 0 || l !== Ut) && (l === Ut && ((gt & 2) === 0 && (Uc |= u), $t === 4 && ju(
      l,
      nt,
      Fa,
      !1
    )), pa(l));
  }
  function Vo(l, n, u) {
    if ((gt & 6) !== 0) throw Error(_(327));
    var c = !u && (n & 124) === 0 && (n & l.expiredLanes) === 0 || m(l, n), r = c ? hm(l, n) : xd(l, n, !0), s = c;
    do {
      if (r === 0) {
        qo && !c && ju(l, n, 0, !1);
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
              r = Yo;
              var S = p.current.memoizedState.isDehydrated;
              if (S && (au(p, y).flags |= 256), y = xd(
                p,
                y,
                !1
              ), y !== 2) {
                if (im && !S) {
                  p.errorRecoveryDisabledLanes |= s, Uc |= s, r = 4;
                  break e;
                }
                s = ma, ma = r, s !== null && (ma === null ? ma = s : ma.push.apply(
                  ma,
                  s
                ));
              }
              r = y;
            }
            if (s = !1, r !== 2) continue;
          }
        }
        if (r === 1) {
          au(l, 0), ju(l, n, 0, !0);
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
              ju(
                c,
                n,
                Fa,
                !eu
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
          if ((n & 62914560) === n && (r = om + 300 - pl(), 10 < r)) {
            if (ju(
              c,
              n,
              Fa,
              !eu
            ), un(c, 0, !0) !== 0) break e;
            c.timeoutHandle = Vd(
              br.bind(
                null,
                c,
                u,
                ma,
                jo,
                cm,
                n,
                Fa,
                Uc,
                Bo,
                eu,
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
            ma,
            jo,
            cm,
            n,
            Fa,
            Uc,
            Bo,
            eu,
            s,
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
  function br(l, n, u, c, r, s, y, p, S, x, Z, $, w, Y) {
    if (l.timeoutHandle = -1, $ = n.subtreeFlags, ($ & 8192 || ($ & 16785408) === 16785408) && (ef = { stylesheets: null, count: 0, unsuspend: Uv }, sv(n), $ = Um(), $ !== null)) {
      l.cancelPendingCommit = $(
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
          w,
          Y
        )
      ), ju(l, s, y, !x);
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
            if (!_l(s(), r)) return !1;
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
  function ju(l, n, u, c) {
    n &= ~tu, n &= ~Uc, l.suspendedLanes |= n, l.pingedLanes &= ~n, c && (l.warmLanes |= n), c = l.expirationTimes;
    for (var r = n; 0 < r; ) {
      var s = 31 - Dl(r), y = 1 << s;
      c[s] = -1, r &= ~y;
    }
    u !== 0 && ct(l, u, n);
  }
  function _c() {
    return (gt & 6) === 0 ? (Ar(0), !1) : !0;
  }
  function zi() {
    if (at !== null) {
      if (St === 0)
        var l = at.return;
      else
        l = at, pn = Uu = null, Js(l), Sc = null, Uo = 0, l = at;
      for (; l !== null; )
        Jy(l.alternate, l), l = l.return;
      at = null;
    }
  }
  function au(l, n) {
    var u = l.timeoutHandle;
    u !== -1 && (l.timeoutHandle = -1, Bg(u)), u = l.cancelPendingCommit, u !== null && (l.cancelPendingCommit = null, u()), zi(), Ut = l, at = u = yn(l.current, null), nt = n, St = 0, ya = null, eu = !1, qo = m(l, n), im = !1, Bo = Fa = tu = Uc = Yu = $t = 0, ma = Yo = null, cm = !1, (n & 8) !== 0 && (n |= n & 32);
    var c = l.entangledLanes;
    if (c !== 0)
      for (l = l.entanglements, c &= n; 0 < c; ) {
        var r = 31 - Dl(c), s = 1 << r;
        n |= l[r], c &= ~s;
      }
    return Bu = n, hn(), u;
  }
  function dm(l, n) {
    Ve = null, O.H = od, n === vi || n === $f ? (n = py(), St = 3) : n === Ys ? (n = py(), St = 4) : St = n === Kt ? 8 : n !== null && typeof n == "object" && typeof n.then == "function" ? 6 : 1, ya = n, at === null && ($t = 1, hr(
      l,
      Oa(n, l.current)
    ));
  }
  function yv() {
    var l = O.H;
    return O.H = od, l === null ? od : l;
  }
  function Cc() {
    var l = O.A;
    return O.A = um, l;
  }
  function xc() {
    $t = 4, eu || (nt & 4194048) !== nt && Ma.current !== null || (qo = !0), (Yu & 134217727) === 0 && (Uc & 134217727) === 0 || Ut === null || ju(
      Ut,
      nt,
      Fa,
      !1
    );
  }
  function xd(l, n, u) {
    var c = gt;
    gt |= 2;
    var r = yv(), s = Cc();
    (Ut !== l || nt !== n) && (jo = null, au(l, n)), n = !1;
    var y = $t;
    e: do
      try {
        if (St !== 0 && at !== null) {
          var p = at, S = ya;
          switch (St) {
            case 8:
              zi(), y = 6;
              break e;
            case 3:
            case 2:
            case 9:
            case 6:
              Ma.current === null && (n = !0);
              var x = St;
              if (St = 0, ya = null, Xo(l, p, S, x), u && qo) {
                y = 0;
                break e;
              }
              break;
            default:
              x = St, St = 0, ya = null, Xo(l, p, S, x);
          }
        }
        Hd(), y = $t;
        break;
      } catch (Z) {
        dm(l, Z);
      }
    while (!0);
    return n && l.shellSuspendCounter++, pn = Uu = null, gt = c, O.H = r, O.A = s, at === null && (Ut = null, nt = 0, hn()), y;
  }
  function Hd() {
    for (; at !== null; ) mm(at);
  }
  function hm(l, n) {
    var u = gt;
    gt |= 2;
    var c = yv(), r = Cc();
    Ut !== l || nt !== n ? (jo = null, Ud = pl() + 500, au(l, n)) : qo = m(
      l,
      n
    );
    e: do
      try {
        if (St !== 0 && at !== null) {
          n = at;
          var s = ya;
          t: switch (St) {
            case 1:
              St = 0, ya = null, Xo(l, n, s, 1);
              break;
            case 2:
            case 9:
              if (Gs(s)) {
                St = 0, ya = null, pm(n);
                break;
              }
              n = function() {
                St !== 2 && St !== 9 || Ut !== l || (St = 7), pa(l);
              }, s.then(n, n);
              break e;
            case 3:
              St = 7;
              break e;
            case 4:
              St = 5;
              break e;
            case 7:
              Gs(s) ? (St = 0, ya = null, pm(n)) : (St = 0, ya = null, Xo(l, n, s, 7));
              break;
            case 5:
              var y = null;
              switch (at.tag) {
                case 26:
                  y = at.memoizedState;
                case 5:
                case 27:
                  var p = at;
                  if (!y || Mm(y)) {
                    St = 0, ya = null;
                    var S = p.sibling;
                    if (S !== null) at = S;
                    else {
                      var x = p.return;
                      x !== null ? (at = x, Sr(x)) : at = null;
                    }
                    break t;
                  }
              }
              St = 0, ya = null, Xo(l, n, s, 5);
              break;
            case 6:
              St = 0, ya = null, Xo(l, n, s, 6);
              break;
            case 8:
              zi(), $t = 6;
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
    return pn = Uu = null, O.H = c, O.A = r, gt = u, at !== null ? 0 : (Ut = null, nt = 0, hn(), $t);
  }
  function ym() {
    for (; at !== null && !Sf(); )
      mm(at);
  }
  function mm(l) {
    var n = fv(l.alternate, l, Bu);
    l.memoizedProps = l.pendingProps, n === null ? Sr(l) : at = n;
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
          nt
        );
        break;
      case 11:
        n = Vy(
          u,
          n,
          n.pendingProps,
          n.type.render,
          n.ref,
          nt
        );
        break;
      case 5:
        Js(n);
      default:
        Jy(u, n), n = at = We(n, Bu), n = fv(u, n, Bu);
    }
    l.memoizedProps = l.pendingProps, n === null ? Sr(l) : at = n;
  }
  function Xo(l, n, u, c) {
    pn = Uu = null, Js(n), Sc = null, Uo = 0;
    var r = n.return;
    try {
      if (iv(
        l,
        r,
        n,
        u,
        nt
      )) {
        $t = 1, hr(
          l,
          Oa(u, l.current)
        ), at = null;
        return;
      }
    } catch (s) {
      if (r !== null) throw at = r, s;
      $t = 1, hr(
        l,
        Oa(u, l.current)
      ), at = null;
      return;
    }
    n.flags & 32768 ? (st || c === 1 ? l = !0 : qo || (nt & 536870912) !== 0 ? l = !1 : (eu = l = !0, (c === 2 || c === 9 || c === 3 || c === 6) && (c = Ma.current, c !== null && c.tag === 13 && (c.flags |= 16384))), mv(n, l)) : Sr(n);
  }
  function Sr(l) {
    var n = l;
    do {
      if ((n.flags & 32768) !== 0) {
        mv(
          n,
          eu
        );
        return;
      }
      l = n.return;
      var u = Ky(
        n.alternate,
        n,
        Bu
      );
      if (u !== null) {
        at = u;
        return;
      }
      if (n = n.sibling, n !== null) {
        at = n;
        return;
      }
      at = n = l;
    } while (n !== null);
    $t === 0 && ($t = 5);
  }
  function mv(l, n) {
    do {
      var u = Cg(l.alternate, l);
      if (u !== null) {
        u.flags &= 32767, at = u;
        return;
      }
      if (u = l.return, u !== null && (u.flags |= 32768, u.subtreeFlags = 0, u.deletions = null), !n && (l = l.sibling, l !== null)) {
        at = l;
        return;
      }
      at = l = u;
    } while (l !== null);
    $t = 6, at = null;
  }
  function pv(l, n, u, c, r, s, y, p, S) {
    l.cancelPendingCommit = null;
    do
      wd();
    while (Hl !== 0);
    if ((gt & 6) !== 0) throw Error(_(327));
    if (n !== null) {
      if (n === l.current) throw Error(_(177));
      if (s = n.lanes | n.childLanes, s |= qn, Le(
        l,
        u,
        s,
        y,
        p,
        S
      ), l === Ut && (at = Ut = null, nt = 0), Go = n, lu = l, Nl = u, _d = s, Cd = r, fm = c, (n.subtreeFlags & 10256) !== 0 || (n.flags & 10256) !== 0 ? (l.callbackNode = null, l.callbackPriority = 0, Ng(Un, function() {
        return vm(), null;
      })) : (l.callbackNode = null, l.callbackPriority = 0), c = (n.flags & 13878) !== 0, (n.subtreeFlags & 13878) !== 0 || c) {
        c = O.T, O.T = null, r = W.p, W.p = 2, y = gt, gt |= 4;
        try {
          Wy(l, n, u);
        } finally {
          gt = y, W.p = r, O.T = c;
        }
      }
      Hl = 1, vv(), Tr(), Nd();
    }
  }
  function vv() {
    if (Hl === 1) {
      Hl = 0;
      var l = lu, n = Go, u = (n.flags & 13878) !== 0;
      if ((n.subtreeFlags & 13878) !== 0 || u) {
        u = O.T, O.T = null;
        var c = W.p;
        W.p = 2;
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
              var x = S.start, Z = S.end;
              if (Z === void 0 && (Z = x), "selectionStart" in p)
                p.selectionStart = x, p.selectionEnd = Math.min(
                  Z,
                  p.value.length
                );
              else {
                var $ = p.ownerDocument || document, w = $ && $.defaultView || window;
                if (w.getSelection) {
                  var Y = w.getSelection(), Ae = p.textContent.length, Re = Math.min(S.start, Ae), yt = S.end === void 0 ? Re : Math.min(S.end, Ae);
                  !Y.extend && Re > yt && (y = yt, yt = Re, Re = y);
                  var z = xt(
                    p,
                    Re
                  ), R = xt(
                    p,
                    yt
                  );
                  if (z && R && (Y.rangeCount !== 1 || Y.anchorNode !== z.node || Y.anchorOffset !== z.offset || Y.focusNode !== R.node || Y.focusOffset !== R.offset)) {
                    var C = $.createRange();
                    C.setStart(z.node, z.offset), Y.removeAllRanges(), Re > yt ? (Y.addRange(C), Y.extend(R.node, R.offset)) : (C.setEnd(R.node, R.offset), Y.addRange(C));
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
              var J = $[p];
              J.element.scrollLeft = J.left, J.element.scrollTop = J.top;
            }
          }
          qr = !!_r, Cr = _r = null;
        } finally {
          gt = r, W.p = c, O.T = u;
        }
      }
      l.current = n, Hl = 2;
    }
  }
  function Tr() {
    if (Hl === 2) {
      Hl = 0;
      var l = lu, n = Go, u = (n.flags & 8772) !== 0;
      if ((n.subtreeFlags & 8772) !== 0 || u) {
        u = O.T, O.T = null;
        var c = W.p;
        W.p = 2;
        var r = gt;
        gt |= 4;
        try {
          Fy(l, n.alternate, n);
        } finally {
          gt = r, W.p = c, O.T = u;
        }
      }
      Hl = 3;
    }
  }
  function Nd() {
    if (Hl === 4 || Hl === 3) {
      Hl = 0, tl();
      var l = lu, n = Go, u = Nl, c = fm;
      (n.subtreeFlags & 10256) !== 0 || (n.flags & 10256) !== 0 ? Hl = 5 : (Hl = 0, Go = lu = null, gv(l, l.pendingLanes));
      var r = l.pendingLanes;
      if (r === 0 && (Di = null), cn(u), n = n.stateNode, Ol && typeof Ol.onCommitFiberRoot == "function")
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
        n = O.T, r = W.p, W.p = 2, O.T = null;
        try {
          for (var s = l.onRecoverableError, y = 0; y < c.length; y++) {
            var p = c[y];
            s(p.value, {
              componentStack: p.stack
            });
          }
        } finally {
          O.T = n, W.p = r;
        }
      }
      (Nl & 3) !== 0 && wd(), pa(l), r = l.pendingLanes, (u & 4194090) !== 0 && (r & 42) !== 0 ? l === rm ? Lo++ : (Lo = 0, rm = l) : Lo = 0, Ar(0);
    }
  }
  function gv(l, n) {
    (l.pooledCacheLanes &= n) === 0 && (n = l.pooledCache, n != null && (l.pooledCache = null, Gn(n)));
  }
  function wd(l) {
    return vv(), Tr(), Nd(), vm();
  }
  function vm() {
    if (Hl !== 5) return !1;
    var l = lu, n = _d;
    _d = 0;
    var u = cn(Nl), c = O.T, r = W.p;
    try {
      W.p = 32 > u ? 32 : u, O.T = null, u = Cd, Cd = null;
      var s = lu, y = Nl;
      if (Hl = 0, Go = lu = null, Nl = 0, (gt & 6) !== 0) throw Error(_(331));
      var p = gt;
      if (gt |= 4, am(s.current), tm(
        s,
        s.current,
        y,
        u
      ), gt = p, Ar(0, !1), Ol && typeof Ol.onPostCommitFiberRoot == "function")
        try {
          Ol.onPostCommitFiberRoot(Pu, s);
        } catch {
        }
      return !0;
    } finally {
      W.p = r, O.T = c, gv(l, n);
    }
  }
  function gm(l, n, u) {
    n = Oa(u, n), n = Gy(l.stateNode, n, 2), l = Xn(l, n, 2), l !== null && (we(l, 2), pa(l));
  }
  function Et(l, n, u) {
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
          if (typeof n.type.getDerivedStateFromError == "function" || typeof c.componentDidCatch == "function" && (Di === null || !Di.has(c))) {
            l = Oa(u, l), u = Ly(2), c = Xn(n, u, 2), c !== null && (ha(
              u,
              c,
              n,
              l
            ), we(c, 2), pa(c));
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
    c !== null && c.delete(n), l.pingedLanes |= l.suspendedLanes & u, l.warmLanes &= ~u, Ut === l && (nt & u) === u && ($t === 4 || $t === 3 && (nt & 62914560) === nt && 300 > pl() - om ? (gt & 2) === 0 && au(l, 0) : tu |= u, Bo === nt && (Bo = 0)), pa(l);
  }
  function Sm(l, n) {
    n === 0 && (n = ue()), l = Bn(l, n), l !== null && (we(l, n), pa(l));
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
    return Mn(l, n);
  }
  var Bd = null, Mi = null, Er = !1, Qo = !1, Yd = !1, Ui = 0;
  function pa(l) {
    l !== Mi && l.next === null && (Mi === null ? Bd = Mi = l : Mi = Mi.next = l), Qo = !0, Er || (Er = !0, Tv());
  }
  function Ar(l, n) {
    if (!Yd && Qo) {
      Yd = !0;
      do
        for (var u = !1, c = Bd; c !== null; ) {
          if (l !== 0) {
            var r = c.pendingLanes;
            if (r === 0) var s = 0;
            else {
              var y = c.suspendedLanes, p = c.pingedLanes;
              s = (1 << 31 - Dl(42 | l) + 1) - 1, s &= r & ~(y & ~p), s = s & 201326741 ? s & 201326741 | 1 : s ? s | 2 : 0;
            }
            s !== 0 && (u = !0, Or(c, s));
          } else
            s = nt, s = un(
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
    Rr();
  }
  function Rr() {
    Qo = Er = !1;
    var l = 0;
    Ui !== 0 && (Vu() && (l = Ui), Ui = 0);
    for (var n = pl(), u = null, c = Bd; c !== null; ) {
      var r = c.next, s = Tm(c, n);
      s === 0 ? (c.next = null, u === null ? Bd = r : u.next = r, r === null && (Mi = u)) : (u = c, (l !== 0 || (s & 3) !== 0) && (Qo = !0)), c = r;
    }
    Ar(l);
  }
  function Tm(l, n) {
    for (var u = l.suspendedLanes, c = l.pingedLanes, r = l.expirationTimes, s = l.pendingLanes & -62914561; 0 < s; ) {
      var y = 31 - Dl(s), p = 1 << y, S = r[y];
      S === -1 ? ((p & u) === 0 || (p & c) !== 0) && (r[y] = D(p, n)) : S <= n && (l.expiredLanes |= p), s &= ~p;
    }
    if (n = Ut, u = nt, u = un(
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
          u = Je;
          break;
        case 32:
          u = Un;
          break;
        case 268435456:
          u = bu;
          break;
        default:
          u = Un;
      }
      return c = Sv.bind(null, l), u = Mn(u, c), l.callbackPriority = n, l.callbackNode = u, n;
    }
    return c !== null && c !== null && Pc(c), l.callbackPriority = 2, l.callbackNode = null, 2;
  }
  function Sv(l, n) {
    if (Hl !== 0 && Hl !== 5)
      return l.callbackNode = null, l.callbackPriority = 0, null;
    var u = l.callbackNode;
    if (wd() && l.callbackNode !== u)
      return null;
    var c = nt;
    return c = un(
      l,
      l === Ut ? c : 0,
      l.cancelPendingCommit !== null || l.timeoutHandle !== -1
    ), c === 0 ? null : (Vo(l, c, n), Tm(l, pl()), l.callbackNode != null && l.callbackNode === u ? Sv.bind(null, l) : null);
  }
  function Or(l, n) {
    if (wd()) return null;
    Vo(l, n, !0);
  }
  function Tv() {
    Yg(function() {
      (gt & 6) !== 0 ? Mn(
        cs,
        bv
      ) : Rr();
    });
  }
  function Hc() {
    return Ui === 0 && (Ui = te()), Ui;
  }
  function jd(l) {
    return l == null || typeof l == "symbol" || typeof l == "boolean" ? null : typeof l == "function" ? l : _f("" + l);
  }
  function Dr(l, n) {
    var u = n.ownerDocument.createElement("input");
    return u.name = n.name, u.value = n.value, l.id && u.setAttribute("form", l.id), n.parentNode.insertBefore(u, n), l = new FormData(l), u.parentNode.removeChild(u), l;
  }
  function Ev(l, n, u, c, r) {
    if (n === "submit" && u && u.stateNode === r) {
      var s = jd(
        (r[kl] || null).action
      ), y = c.submitter;
      y && (n = (n = y[kl] || null) ? jd(n.formAction) : y.getAttribute("formAction"), n !== null && (s = n, y = null));
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
  for (var Wt = 0; Wt < so.length; Wt++) {
    var zr = so[Wt], wg = zr.toLowerCase(), ke = zr[0].toUpperCase() + zr.slice(1);
    La(
      wg,
      "on" + ke
    );
  }
  La(Zp, "onAnimationEnd"), La(iy, "onAnimationIteration"), La(Kp, "onAnimationStart"), La("dblclick", "onDoubleClick"), La("focusin", "onFocus"), La("focusout", "onBlur"), La(cy, "onTransitionRun"), La(_s, "onTransitionStart"), La(Jp, "onTransitionCancel"), La(oy, "onTransitionEnd"), ti("onMouseEnter", ["mouseout", "mouseover"]), ti("onMouseLeave", ["mouseout", "mouseover"]), ti("onPointerEnter", ["pointerout", "pointerover"]), ti("onPointerLeave", ["pointerout", "pointerover"]), ei(
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
  var Mr = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(
    " "
  ), _i = new Set(
    "beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(Mr)
  );
  function Nc(l, n) {
    n = (n & 4) !== 0;
    for (var u = 0; u < l.length; u++) {
      var c = l[u], r = c.event;
      c = c.listeners;
      e: {
        var s = void 0;
        if (n)
          for (var y = c.length - 1; 0 <= y; y--) {
            var p = c[y], S = p.instance, x = p.currentTarget;
            if (p = p.listener, S !== s && r.isPropagationStopped())
              break e;
            s = p, r.currentTarget = x;
            try {
              s(r);
            } catch (Z) {
              sr(Z);
            }
            r.currentTarget = null, s = S;
          }
        else
          for (y = 0; y < c.length; y++) {
            if (p = c[y], S = p.instance, x = p.currentTarget, p = p.listener, S !== s && r.isPropagationStopped())
              break e;
            s = p, r.currentTarget = x;
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
  function Xe(l, n) {
    var u = n[rs];
    u === void 0 && (u = n[rs] = /* @__PURE__ */ new Set());
    var c = l + "__bubble";
    u.has(c) || (Gd(n, l, 2, !1), u.add(c));
  }
  function Zo(l, n, u) {
    var c = 0;
    n && (c |= 4), Gd(
      u,
      l,
      c,
      n
    );
  }
  var Ko = "_reactListening" + Math.random().toString(36).slice(2);
  function Em(l) {
    if (!l[Ko]) {
      l[Ko] = !0, Of.forEach(function(u) {
        u !== "selectionchange" && (_i.has(u) || Zo(u, !1, l), Zo(u, !0, l));
      });
      var n = l.nodeType === 9 ? l : l.ownerDocument;
      n === null || n[Ko] || (n[Ko] = !0, Zo("selectionchange", !1, n));
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
  function Ia(l, n, u, c, r) {
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
            if (y = Ml(p), y === null) return;
            if (S = y.tag, S === 5 || S === 6 || S === 26 || S === 27) {
              c = s = y;
              continue e;
            }
            p = p.parentNode;
          }
        }
        c = c.return;
      }
    oo(function() {
      var x = s, Z = vs(u), $ = [];
      e: {
        var w = fy.get(l);
        if (w !== void 0) {
          var Y = Ts, Ae = l;
          switch (l) {
            case "keypress":
              if (Ul(u) === 0) break e;
            case "keydown":
            case "keyup":
              Y = on;
              break;
            case "focusin":
              Ae = "focus", Y = Qh;
              break;
            case "focusout":
              Ae = "blur", Y = Qh;
              break;
            case "beforeblur":
            case "afterblur":
              Y = Qh;
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
              Y = Xh;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              Y = qp;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              Y = Jh;
              break;
            case Zp:
            case iy:
            case Kp:
              Y = zg;
              break;
            case oy:
              Y = Lp;
              break;
            case "scroll":
            case "scrollend":
              Y = Np;
              break;
            case "wheel":
              Y = ac;
              break;
            case "copy":
            case "cut":
            case "paste":
              Y = Hf;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              Y = Nf;
              break;
            case "toggle":
            case "beforetoggle":
              Y = Vp;
          }
          var Re = (n & 4) !== 0, yt = !Re && (l === "scroll" || l === "scrollend"), z = Re ? w !== null ? w + "Capture" : null : w;
          Re = [];
          for (var R = x, C; R !== null; ) {
            var J = R;
            if (C = J.stateNode, J = J.tag, J !== 5 && J !== 26 && J !== 27 || C === null || z === null || (J = Pi(R, z), J != null && Re.push(
              Gu(R, J, C)
            )), yt) break;
            R = R.return;
          }
          0 < Re.length && (w = new Y(
            w,
            Ae,
            null,
            u,
            Z
          ), $.push({ event: w, listeners: Re }));
        }
      }
      if ((n & 7) === 0) {
        e: {
          if (w = l === "mouseover" || l === "pointerover", Y = l === "mouseout" || l === "pointerout", w && u !== Ii && (Ae = u.relatedTarget || u.fromElement) && (Ml(Ae) || Ae[ao]))
            break e;
          if ((Y || w) && (w = Z.window === Z ? Z : (w = Z.ownerDocument) ? w.defaultView || w.parentWindow : window, Y ? (Ae = u.relatedTarget || u.toElement, Y = x, Ae = Ae ? Ml(Ae) : null, Ae !== null && (yt = he(Ae), Re = Ae.tag, Ae !== yt || Re !== 5 && Re !== 27 && Re !== 6) && (Ae = null)) : (Y = null, Ae = x), Y !== Ae)) {
            if (Re = Xh, J = "onMouseLeave", z = "onMouseEnter", R = "mouse", (l === "pointerout" || l === "pointerover") && (Re = Nf, J = "onPointerLeave", z = "onPointerEnter", R = "pointer"), yt = Y == null ? w : Rf(Y), C = Ae == null ? w : Rf(Ae), w = new Re(
              J,
              R + "leave",
              Y,
              u,
              Z
            ), w.target = yt, w.relatedTarget = C, J = null, Ml(Z) === x && (Re = new Re(
              z,
              R + "enter",
              Ae,
              u,
              Z
            ), Re.target = C, Re.relatedTarget = yt, J = Re), yt = J, Y && Ae)
              t: {
                for (Re = Y, z = Ae, R = 0, C = Re; C; C = Ci(C))
                  R++;
                for (C = 0, J = z; J; J = Ci(J))
                  C++;
                for (; 0 < R - C; )
                  Re = Ci(Re), R--;
                for (; 0 < C - R; )
                  z = Ci(z), C--;
                for (; R--; ) {
                  if (Re === z || z !== null && Re === z.alternate)
                    break t;
                  Re = Ci(Re), z = Ci(z);
                }
                Re = null;
              }
            else Re = null;
            Y !== null && Ur(
              $,
              w,
              Y,
              Re,
              !1
            ), Ae !== null && yt !== null && Ur(
              $,
              yt,
              Ae,
              Re,
              !0
            );
          }
        }
        e: {
          if (w = x ? Rf(x) : window, Y = w.nodeName && w.nodeName.toLowerCase(), Y === "select" || Y === "input" && w.type === "file")
            var se = Ph;
          else if (Ds(w))
            if (ey)
              se = ay;
            else {
              se = ci;
              var Fe = Ms;
            }
          else
            Y = w.nodeName, !Y || Y.toLowerCase() !== "input" || w.type !== "checkbox" && w.type !== "radio" ? x && Fi(x.elementType) && (se = Ph) : se = Ou;
          if (se && (se = se(l, x))) {
            zs(
              $,
              se,
              u,
              Z
            );
            break e;
          }
          Fe && Fe(l, w, x), l === "focusout" && x && w.type === "number" && x.memoizedProps.value != null && Mf(w, "number", w.value);
        }
        switch (Fe = x ? Rf(x) : window, l) {
          case "focusin":
            (Ds(Fe) || Fe.contentEditable === "true") && (Nn = Fe, sn = x, ri = null);
            break;
          case "focusout":
            ri = sn = Nn = null;
            break;
          case "mousedown":
            oc = !0;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            oc = !1, Us($, u, Z);
            break;
          case "selectionchange":
            if (cc) break;
          case "keydown":
          case "keyup":
            Us($, u, Z);
        }
        var Ee;
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
          ii ? Bf(l, u) && (_e = "onCompositionEnd") : l === "keydown" && u.keyCode === 229 && (_e = "onCompositionStart");
        _e && (xn && u.locale !== "ko" && (ii || _e !== "onCompositionStart" ? _e === "onCompositionEnd" && ii && (Ee = Lh()) : (Au = Z, fo = "value" in Au ? Au.value : Au.textContent, ii = !0)), Fe = Jo(x, _e), 0 < Fe.length && (_e = new Zh(
          _e,
          l,
          null,
          u,
          Z
        ), $.push({ event: _e, listeners: Fe }), Ee ? _e.data = Ee : (Ee = ui(u), Ee !== null && (_e.data = Ee)))), (Ee = $h ? Fh(l, u) : nc(l, u)) && (_e = Jo(x, "onBeforeInput"), 0 < _e.length && (Fe = new Zh(
          "onBeforeInput",
          "beforeinput",
          null,
          u,
          Z
        ), $.push({
          event: Fe,
          listeners: _e
        }), Fe.data = Ee)), Ev(
          $,
          l,
          x,
          u,
          Z
        );
      }
      Nc($, n);
    });
  }
  function Gu(l, n, u) {
    return {
      instance: l,
      listener: n,
      currentTarget: u
    };
  }
  function Jo(l, n) {
    for (var u = n + "Capture", c = []; l !== null; ) {
      var r = l, s = r.stateNode;
      if (r = r.tag, r !== 5 && r !== 26 && r !== 27 || s === null || (r = Pi(l, u), r != null && c.unshift(
        Gu(l, r, s)
      ), r = Pi(l, n), r != null && c.push(
        Gu(l, r, s)
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
  function Ur(l, n, u, c, r) {
    for (var s = n._reactName, y = []; u !== null && u !== c; ) {
      var p = u, S = p.alternate, x = p.stateNode;
      if (p = p.tag, S !== null && S === c) break;
      p !== 5 && p !== 26 && p !== 27 || x === null || (S = x, r ? (x = Pi(u, s), x != null && y.unshift(
        Gu(u, x, S)
      )) : r || (x = Pi(u, s), x != null && y.push(
        Gu(u, x, S)
      ))), u = u.return;
    }
    y.length !== 0 && l.push({ event: n, listeners: y });
  }
  var Ca = /\r\n?/g, Am = /\u0000|\uFFFD/g;
  function Av(l) {
    return (typeof l == "string" ? l : "" + l).replace(Ca, `
`).replace(Am, "");
  }
  function Rm(l, n) {
    return n = Av(n), Av(l) === n;
  }
  function Ld() {
  }
  function qe(l, n, u, c, r, s) {
    switch (u) {
      case "children":
        typeof c == "string" ? n === "body" || n === "textarea" && c === "" || uo(l, c) : (typeof c == "number" || typeof c == "bigint") && n !== "body" && uo(l, "" + c);
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
        Uf(l, c, s);
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
        c = _f("" + c), l.setAttribute(u, c);
        break;
      case "onClick":
        c != null && (l.onclick = Ld);
        break;
      case "onScroll":
        c != null && Xe("scroll", l);
        break;
      case "onScrollEnd":
        c != null && Xe("scrollend", l);
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
        Xe("beforetoggle", l), Xe("toggle", l), Tu(l, "popover", c);
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
        Tu(l, "is", c);
        break;
      case "innerText":
      case "textContent":
        break;
      default:
        (!(2 < u.length) || u[0] !== "o" && u[0] !== "O" || u[1] !== "n" && u[1] !== "N") && (u = Rg.get(u) || u, Tu(l, u, c));
    }
  }
  function j(l, n, u, c, r, s) {
    switch (u) {
      case "style":
        Uf(l, c, s);
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
        typeof c == "string" ? uo(l, c) : (typeof c == "number" || typeof c == "bigint") && uo(l, "" + c);
        break;
      case "onScroll":
        c != null && Xe("scroll", l);
        break;
      case "onScrollEnd":
        c != null && Xe("scrollend", l);
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
            if (u[0] === "o" && u[1] === "n" && (r = u.endsWith("Capture"), n = u.slice(2, r ? u.length - 7 : void 0), s = l[kl] || null, s = s != null ? s[u] : null, typeof s == "function" && l.removeEventListener(n, s, r), typeof c == "function")) {
              typeof s != "function" && s !== null && (u in l ? l[u] = null : l.hasAttribute(u) && l.removeAttribute(u)), l.addEventListener(n, c, r);
              break e;
            }
            u in l ? l[u] = c : c === !0 ? l.setAttribute(u, "") : Tu(l, u, c);
          }
    }
  }
  function xe(l, n, u) {
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
        Xe("error", l), Xe("load", l);
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
        Xe("invalid", l);
        var p = s = y = r = null, S = null, x = null;
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
                  x = Z;
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
          x,
          y,
          r,
          !1
        ), ai(l);
        return;
      case "select":
        Xe("invalid", l), c = y = s = null;
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
        n = s, u = y, l.multiple = !!c, n != null ? Wi(l, !!c, n, !1) : u != null && Wi(l, !!c, u, !0);
        return;
      case "textarea":
        Xe("invalid", l), s = r = c = null;
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
        jh(l, c, r, s), ai(l);
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
        Xe("beforetoggle", l), Xe("toggle", l), Xe("cancel", l), Xe("close", l);
        break;
      case "iframe":
      case "object":
        Xe("load", l);
        break;
      case "video":
      case "audio":
        for (c = 0; c < Mr.length; c++)
          Xe(Mr[c], l);
        break;
      case "image":
        Xe("error", l), Xe("load", l);
        break;
      case "details":
        Xe("toggle", l);
        break;
      case "embed":
      case "source":
      case "link":
        Xe("error", l), Xe("load", l);
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
        for (x in u)
          if (u.hasOwnProperty(x) && (c = u[x], c != null))
            switch (x) {
              case "children":
              case "dangerouslySetInnerHTML":
                throw Error(_(137, n));
              default:
                qe(l, n, x, c, u, null);
            }
        return;
      default:
        if (Fi(n)) {
          for (Z in u)
            u.hasOwnProperty(Z) && (c = u[Z], c !== void 0 && j(
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
        var r = null, s = null, y = null, p = null, S = null, x = null, Z = null;
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
                c.hasOwnProperty(Y) || qe(l, n, Y, null, c, $);
            }
        }
        for (var w in c) {
          var Y = c[w];
          if ($ = u[w], c.hasOwnProperty(w) && (Y != null || $ != null))
            switch (w) {
              case "type":
                s = Y;
                break;
              case "name":
                r = Y;
                break;
              case "checked":
                x = Y;
                break;
              case "defaultChecked":
                Z = Y;
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
                Y !== $ && qe(
                  l,
                  n,
                  w,
                  Y,
                  c,
                  $
                );
            }
        }
        ys(
          l,
          y,
          p,
          S,
          x,
          Z,
          s,
          r
        );
        return;
      case "select":
        Y = y = p = w = null;
        for (s in u)
          if (S = u[s], u.hasOwnProperty(s) && S != null)
            switch (s) {
              case "value":
                break;
              case "multiple":
                Y = S;
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
                w = s;
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
        n = p, u = y, c = Y, w != null ? Wi(l, !!u, w, !1) : !!c != !!u && (n != null ? Wi(l, !!u, n, !0) : Wi(l, !!u, u ? [] : "", !1));
        return;
      case "textarea":
        Y = w = null;
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
                w = r;
                break;
              case "defaultValue":
                Y = r;
                break;
              case "children":
                break;
              case "dangerouslySetInnerHTML":
                if (r != null) throw Error(_(91));
                break;
              default:
                r !== s && qe(l, n, y, r, c, s);
            }
        Yh(l, w, Y);
        return;
      case "option":
        for (var Ae in u)
          if (w = u[Ae], u.hasOwnProperty(Ae) && w != null && !c.hasOwnProperty(Ae))
            switch (Ae) {
              case "selected":
                l.selected = !1;
                break;
              default:
                qe(
                  l,
                  n,
                  Ae,
                  null,
                  c,
                  w
                );
            }
        for (S in c)
          if (w = c[S], Y = u[S], c.hasOwnProperty(S) && w !== Y && (w != null || Y != null))
            switch (S) {
              case "selected":
                l.selected = w && typeof w != "function" && typeof w != "symbol";
                break;
              default:
                qe(
                  l,
                  n,
                  S,
                  w,
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
          w = u[Re], u.hasOwnProperty(Re) && w != null && !c.hasOwnProperty(Re) && qe(l, n, Re, null, c, w);
        for (x in c)
          if (w = c[x], Y = u[x], c.hasOwnProperty(x) && w !== Y && (w != null || Y != null))
            switch (x) {
              case "children":
              case "dangerouslySetInnerHTML":
                if (w != null)
                  throw Error(_(137, n));
                break;
              default:
                qe(
                  l,
                  n,
                  x,
                  w,
                  c,
                  Y
                );
            }
        return;
      default:
        if (Fi(n)) {
          for (var yt in u)
            w = u[yt], u.hasOwnProperty(yt) && w !== void 0 && !c.hasOwnProperty(yt) && j(
              l,
              n,
              yt,
              void 0,
              c,
              w
            );
          for (Z in c)
            w = c[Z], Y = u[Z], !c.hasOwnProperty(Z) || w === Y || w === void 0 && Y === void 0 || j(
              l,
              n,
              Z,
              w,
              c,
              Y
            );
          return;
        }
    }
    for (var z in u)
      w = u[z], u.hasOwnProperty(z) && w != null && !c.hasOwnProperty(z) && qe(l, n, z, null, c, w);
    for ($ in c)
      w = c[$], Y = u[$], !c.hasOwnProperty($) || w === Y || w == null && Y == null || qe(l, n, $, w, c, Y);
  }
  var _r = null, Cr = null;
  function Pa(l) {
    return l.nodeType === 9 ? l : l.ownerDocument;
  }
  function Lu(l) {
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
  function nu(l, n) {
    return l === "textarea" || l === "noscript" || typeof n.children == "string" || typeof n.children == "number" || typeof n.children == "bigint" || typeof n.dangerouslySetInnerHTML == "object" && n.dangerouslySetInnerHTML !== null && n.dangerouslySetInnerHTML.__html != null;
  }
  var $o = null;
  function Vu() {
    var l = window.event;
    return l && l.type === "popstate" ? l === $o ? !1 : ($o = l, !0) : ($o = null, !1);
  }
  var Vd = typeof setTimeout == "function" ? setTimeout : void 0, Bg = typeof clearTimeout == "function" ? clearTimeout : void 0, Rv = typeof Promise == "function" ? Promise : void 0, Yg = typeof queueMicrotask == "function" ? queueMicrotask : typeof Rv < "u" ? function(l) {
    return Rv.resolve(null).then(l).catch(uu);
  } : Vd;
  function uu(l) {
    setTimeout(function() {
      throw l;
    });
  }
  function xi(l) {
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
            if (u & 1 && va(y.documentElement), u & 2 && va(y.body), u & 4)
              for (u = y.head, va(u), y = u.firstChild; y; ) {
                var p = y.nextSibling, S = y.nodeName;
                y[ye] || S === "SCRIPT" || S === "STYLE" || S === "LINK" && y.rel.toLowerCase() === "stylesheet" || u.removeChild(y), y = p;
              }
          }
          if (r === 0) {
            l.removeChild(s), cu(n);
            return;
          }
          r--;
        } else
          u === "$" || u === "$?" || u === "$!" ? r++ : c = u.charCodeAt(0) - 48;
      else c = 0;
      u = s;
    } while (u);
    cu(n);
  }
  function xr(l) {
    var n = l.firstChild;
    for (n && n.nodeType === 10 && (n = n.nextSibling); n; ) {
      var u = n;
      switch (n = n.nextSibling, u.nodeName) {
        case "HTML":
        case "HEAD":
        case "BODY":
          xr(u), Af(u);
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
      var r = u;
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
      if (l = Tn(l.nextSibling), l === null) break;
    }
    return null;
  }
  function jg(l, n, u) {
    if (n === "") return null;
    for (; l.nodeType !== 3; )
      if ((l.nodeType !== 1 || l.nodeName !== "INPUT" || l.type !== "hidden") && !u || (l = Tn(l.nextSibling), l === null)) return null;
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
    Af(l);
  }
  var Ft = /* @__PURE__ */ new Map(), Ql = /* @__PURE__ */ new Set();
  function Qd(l) {
    return typeof l.getRootNode == "function" ? l.getRootNode() : l.nodeType === 9 ? l : l.ownerDocument;
  }
  var Xu = W.d;
  W.d = {
    f: Zd,
    r: Kd,
    D: Qu,
    C: Jd,
    L: Ni,
    m: Zl,
    X: wi,
    S: ga,
    M: Om
  };
  function Zd() {
    var l = Xu.f(), n = _c();
    return l || n;
  }
  function Kd(l) {
    var n = Ki(l);
    n !== null && n.tag === 5 && n.type === "form" ? Mo(n) : Xu.r(l);
  }
  var ql = typeof document > "u" ? null : document;
  function En(l, n, u) {
    var c = ql;
    if (c && typeof n == "string" && n) {
      var r = ja(n);
      r = 'link[rel="' + l + '"][href="' + r + '"]', typeof u == "string" && (r += '[crossorigin="' + u + '"]'), Ql.has(r) || (Ql.add(r), l = { rel: l, crossOrigin: u, href: n }, c.querySelector(r) === null && (n = c.createElement("link"), xe(n, "link", l), ol(n), c.head.appendChild(n)));
    }
  }
  function Qu(l) {
    Xu.D(l), En("dns-prefetch", l, null);
  }
  function Jd(l, n) {
    Xu.C(l, n), En("preconnect", l, n);
  }
  function Ni(l, n, u) {
    Xu.L(l, n, u);
    var c = ql;
    if (c && l && n) {
      var r = 'link[rel="preload"][as="' + ja(n) + '"]';
      n === "image" && u && u.imageSrcSet ? (r += '[imagesrcset="' + ja(
        u.imageSrcSet
      ) + '"]', typeof u.imageSizes == "string" && (r += '[imagesizes="' + ja(
        u.imageSizes
      ) + '"]')) : r += '[href="' + ja(l) + '"]';
      var s = r;
      switch (n) {
        case "style":
          s = Fo(l);
          break;
        case "script":
          s = en(l);
      }
      Ft.has(s) || (l = le(
        {
          rel: "preload",
          href: n === "image" && u && u.imageSrcSet ? void 0 : l,
          as: n
        },
        u
      ), Ft.set(s, l), c.querySelector(r) !== null || n === "style" && c.querySelector(Io(s)) || n === "script" && c.querySelector(wc(s)) || (n = c.createElement("link"), xe(n, "link", l), ol(n), c.head.appendChild(n)));
    }
  }
  function Zl(l, n) {
    Xu.m(l, n);
    var u = ql;
    if (u && l) {
      var c = n && typeof n.as == "string" ? n.as : "script", r = 'link[rel="modulepreload"][as="' + ja(c) + '"][href="' + ja(l) + '"]', s = r;
      switch (c) {
        case "audioworklet":
        case "paintworklet":
        case "serviceworker":
        case "sharedworker":
        case "worker":
        case "script":
          s = en(l);
      }
      if (!Ft.has(s) && (l = le({ rel: "modulepreload", href: l }, n), Ft.set(s, l), u.querySelector(r) === null)) {
        switch (c) {
          case "audioworklet":
          case "paintworklet":
          case "serviceworker":
          case "sharedworker":
          case "worker":
          case "script":
            if (u.querySelector(wc(s)))
              return;
        }
        c = u.createElement("link"), xe(c, "link", l), ol(c), u.head.appendChild(c);
      }
    }
  }
  function ga(l, n, u) {
    Xu.S(l, n, u);
    var c = ql;
    if (c && l) {
      var r = Su(c).hoistableStyles, s = Fo(l);
      n = n || "default";
      var y = r.get(s);
      if (!y) {
        var p = { loading: 0, preload: null };
        if (y = c.querySelector(
          Io(s)
        ))
          p.loading = 5;
        else {
          l = le(
            { rel: "stylesheet", href: l, "data-precedence": n },
            u
          ), (u = Ft.get(s)) && $d(l, u);
          var S = y = c.createElement("link");
          ol(S), xe(S, "link", l), S._p = new Promise(function(x, Z) {
            S.onload = x, S.onerror = Z;
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
  function wi(l, n) {
    Xu.X(l, n);
    var u = ql;
    if (u && l) {
      var c = Su(u).hoistableScripts, r = en(l), s = c.get(r);
      s || (s = u.querySelector(wc(r)), s || (l = le({ src: l, async: !0 }, n), (n = Ft.get(r)) && Wd(l, n), s = u.createElement("script"), ol(s), xe(s, "link", l), u.head.appendChild(s)), s = {
        type: "script",
        instance: s,
        count: 1,
        state: null
      }, c.set(r, s));
    }
  }
  function Om(l, n) {
    Xu.M(l, n);
    var u = ql;
    if (u && l) {
      var c = Su(u).hoistableScripts, r = en(l), s = c.get(r);
      s || (s = u.querySelector(wc(r)), s || (l = le({ src: l, async: !0, type: "module" }, n), (n = Ft.get(r)) && Wd(l, n), s = u.createElement("script"), ol(s), xe(s, "link", l), u.head.appendChild(s)), s = {
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
        return typeof u.precedence == "string" && typeof u.href == "string" ? (n = Fo(u.href), u = Su(
          r
        ).hoistableStyles, c = u.get(n), c || (c = {
          type: "style",
          instance: null,
          count: 0,
          state: null
        }, u.set(n, c)), c) : { type: "void", instance: null, count: 0, state: null };
      case "link":
        if (u.rel === "stylesheet" && typeof u.href == "string" && typeof u.precedence == "string") {
          l = Fo(u.href);
          var s = Su(
            r
          ).hoistableStyles, y = s.get(l);
          if (y || (r = r.ownerDocument || r, y = {
            type: "stylesheet",
            instance: null,
            count: 0,
            state: { loading: 0, preload: null }
          }, s.set(l, y), (s = r.querySelector(
            Io(l)
          )) && !s._p && (y.instance = s, y.state.loading = 5), Ft.has(l) || (u = {
            rel: "preload",
            as: "style",
            href: u.href,
            crossOrigin: u.crossOrigin,
            integrity: u.integrity,
            media: u.media,
            hrefLang: u.hrefLang,
            referrerPolicy: u.referrerPolicy
          }, Ft.set(l, u), s || Dv(
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
        return n = u.async, u = u.src, typeof u == "string" && n && typeof n != "function" && typeof n != "symbol" ? (n = en(u), u = Su(
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
  function Fo(l) {
    return 'href="' + ja(l) + '"';
  }
  function Io(l) {
    return 'link[rel="stylesheet"][' + l + "]";
  }
  function Po(l) {
    return le({}, l, {
      "data-precedence": l.precedence,
      precedence: null
    });
  }
  function Dv(l, n, u, c) {
    l.querySelector('link[rel="preload"][as="style"][' + n + "]") ? c.loading = 1 : (n = l.createElement("link"), c.preload = n, n.addEventListener("load", function() {
      return c.loading |= 1;
    }), n.addEventListener("error", function() {
      return c.loading |= 2;
    }), xe(n, "link", u), ol(n), l.head.appendChild(n));
  }
  function en(l) {
    return '[src="' + ja(l) + '"]';
  }
  function wc(l) {
    return "script[async]" + l;
  }
  function zv(l, n, u) {
    if (n.count++, n.instance === null)
      switch (n.type) {
        case "style":
          var c = l.querySelector(
            'style[data-href~="' + ja(u.href) + '"]'
          );
          if (c)
            return n.instance = c, ol(c), c;
          var r = le({}, u, {
            "data-href": u.href,
            "data-precedence": u.precedence,
            href: null,
            precedence: null
          });
          return c = (l.ownerDocument || l).createElement(
            "style"
          ), ol(c), xe(c, "style", r), kd(c, u.precedence, l), n.instance = c;
        case "stylesheet":
          r = Fo(u.href);
          var s = l.querySelector(
            Io(r)
          );
          if (s)
            return n.state.loading |= 4, n.instance = s, ol(s), s;
          c = Po(u), (r = Ft.get(r)) && $d(c, r), s = (l.ownerDocument || l).createElement("link"), ol(s);
          var y = s;
          return y._p = new Promise(function(p, S) {
            y.onload = p, y.onerror = S;
          }), xe(s, "link", c), n.state.loading |= 4, kd(s, u.precedence, l), n.instance = s;
        case "script":
          return s = en(u.src), (r = l.querySelector(
            wc(s)
          )) ? (n.instance = r, ol(r), r) : (c = u, (r = Ft.get(s)) && (c = le({}, u), Wd(c, r)), l = l.ownerDocument || l, r = l.createElement("script"), ol(r), xe(r, "link", c), l.head.appendChild(r), n.instance = r);
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
  var qi = null;
  function Dm(l, n, u) {
    if (qi === null) {
      var c = /* @__PURE__ */ new Map(), r = qi = /* @__PURE__ */ new Map();
      r.set(u, c);
    } else
      r = qi, c = r.get(u), c || (c = /* @__PURE__ */ new Map(), r.set(u, c));
    if (c.has(l)) return c;
    for (c.set(l, null), u = u.getElementsByTagName(l), r = 0; r < u.length; r++) {
      var s = u[r];
      if (!(s[ye] || s[vl] || l === "link" && s.getAttribute("rel") === "stylesheet") && s.namespaceURI !== "http://www.w3.org/2000/svg") {
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
  var ef = null;
  function Uv() {
  }
  function _v(l, n, u) {
    if (ef === null) throw Error(_(475));
    var c = ef;
    if (n.type === "stylesheet" && (typeof u.media != "string" || matchMedia(u.media).matches !== !1) && (n.state.loading & 4) === 0) {
      if (n.instance === null) {
        var r = Fo(u.href), s = l.querySelector(
          Io(r)
        );
        if (s) {
          l = s._p, l !== null && typeof l == "object" && typeof l.then == "function" && (c.count++, c = Nr.bind(c), l.then(c, c)), n.state.loading |= 4, n.instance = s, ol(s);
          return;
        }
        s = l.ownerDocument || l, u = Po(u), (r = Ft.get(r)) && $d(u, r), s = s.createElement("link"), ol(s);
        var y = s;
        y._p = new Promise(function(p, S) {
          y.onload = p, y.onerror = S;
        }), xe(s, "link", u), n.instance = s;
      }
      c.stylesheets === null && (c.stylesheets = /* @__PURE__ */ new Map()), c.stylesheets.set(n, l), (l = n.state.preload) && (n.state.loading & 3) === 0 && (c.count++, n = Nr.bind(c), l.addEventListener("load", n), l.addEventListener("error", n));
    }
  }
  function Um() {
    if (ef === null) throw Error(_(475));
    var l = ef;
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
  var tf = null;
  function wr(l, n) {
    l.stylesheets = null, l.unsuspend !== null && (l.count++, tf = /* @__PURE__ */ new Map(), n.forEach(xa, l), tf = null, Nr.call(l));
  }
  function xa(l, n) {
    if (!(n.state.loading & 4)) {
      var u = tf.get(l);
      if (u) var c = u.get(null);
      else {
        u = /* @__PURE__ */ new Map(), tf.set(l, u);
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
  var ba = {
    $$typeof: Me,
    Provider: null,
    Consumer: null,
    _currentValue: P,
    _currentValue2: P,
    _threadCount: 0
  };
  function Lg(l, n, u, c, r, s, y, p) {
    this.tag = 1, this.containerInfo = l, this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.next = this.pendingContext = this.context = this.cancelPendingCommit = null, this.callbackPriority = 0, this.expirationTimes = ve(-1), this.entangledLanes = this.shellSuspendCounter = this.errorRecoveryDisabledLanes = this.expiredLanes = this.warmLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = ve(0), this.hiddenUpdates = ve(null), this.identifierPrefix = c, this.onUncaughtError = r, this.onCaughtError = s, this.onRecoverableError = y, this.pooledCache = null, this.pooledCacheLanes = 0, this.formState = p, this.incompleteTransitions = /* @__PURE__ */ new Map();
  }
  function _m(l, n, u, c, r, s, y, p, S, x, Z, $) {
    return l = new Lg(
      l,
      n,
      u,
      y,
      p,
      S,
      x,
      $
    ), n = 1, s === !0 && (n |= 24), s = oa(3, null, null, n), l.current = s, s.stateNode = l, n = Ao(), n.refCount++, l.pooledCache = n, n.refCount++, s.memoizedState = {
      element: c,
      isDehydrated: u,
      cache: n
    }, Ls(s), l;
  }
  function Cm(l) {
    return l ? (l = mo, l) : mo;
  }
  function xm(l, n, u, c, r, s) {
    r = Cm(r), c.context === null ? c.context = r : c.pendingContext = r, c = ra(n), c.payload = { element: u }, s = s === void 0 ? null : s, s !== null && (c.callback = s), u = Xn(l, c, n), u !== null && (_a(u, l, n), yc(u, l, n));
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
      var n = Bn(l, 67108864);
      n !== null && _a(n, l, 67108864), Fd(l, 67108864);
    }
  }
  var qr = !0;
  function Cv(l, n, u, c) {
    var r = O.T;
    O.T = null;
    var s = W.p;
    try {
      W.p = 2, wm(l, n, u, c);
    } finally {
      W.p = s, O.T = r;
    }
  }
  function xv(l, n, u, c) {
    var r = O.T;
    O.T = null;
    var s = W.p;
    try {
      W.p = 8, wm(l, n, u, c);
    } finally {
      W.p = s, O.T = r;
    }
  }
  function wm(l, n, u, c) {
    if (qr) {
      var r = Id(c);
      if (r === null)
        Ia(
          l,
          n,
          c,
          Pd,
          u
        ), qc(l, c);
      else if (Nv(
        r,
        l,
        n,
        u,
        c
      ))
        c.stopPropagation();
      else if (qc(l, c), n & 4 && -1 < Hv.indexOf(l)) {
        for (; r !== null; ) {
          var s = Ki(r);
          if (s !== null)
            switch (s.tag) {
              case 3:
                if (s = s.stateNode, s.current.memoizedState.isDehydrated) {
                  var y = zl(s.pendingLanes);
                  if (y !== 0) {
                    var p = s;
                    for (p.pendingLanes |= 2, p.entangledLanes |= 2; y; ) {
                      var S = 1 << 31 - Dl(y);
                      p.entanglements[1] |= S, y &= ~S;
                    }
                    pa(s), (gt & 6) === 0 && (Ud = pl() + 500, Ar(0));
                  }
                }
                break;
              case 13:
                p = Bn(s, 2), p !== null && _a(p, s, 2), _c(), Fd(s, 2);
            }
          if (s = Id(c), s === null && Ia(
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
        Ia(
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
    if (Pd = null, l = Ml(l), l !== null) {
      var n = he(l);
      if (n === null) l = null;
      else {
        var u = n.tag;
        if (u === 13) {
          if (l = Oe(n), l !== null) return l;
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
        switch (Iu()) {
          case cs:
            return 2;
          case Je:
            return 8;
          case Un:
          case eo:
            return 32;
          case bu:
            return 268435456;
          default:
            return 32;
        }
      default:
        return 32;
    }
  }
  var lf = !1, iu = null, Zu = null, Ku = null, Br = /* @__PURE__ */ new Map(), Yr = /* @__PURE__ */ new Map(), Bi = [], Hv = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(
    " "
  );
  function qc(l, n) {
    switch (l) {
      case "focusin":
      case "focusout":
        iu = null;
        break;
      case "dragenter":
      case "dragleave":
        Zu = null;
        break;
      case "mouseover":
      case "mouseout":
        Ku = null;
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
  function Bc(l, n, u, c, r, s) {
    return l === null || l.nativeEvent !== s ? (l = {
      blockedOn: n,
      domEventName: u,
      eventSystemFlags: c,
      nativeEvent: s,
      targetContainers: [r]
    }, n !== null && (n = Ki(n), n !== null && Nm(n)), l) : (l.eventSystemFlags |= c, n = l.targetContainers, r !== null && n.indexOf(r) === -1 && n.push(r), l);
  }
  function Nv(l, n, u, c, r) {
    switch (n) {
      case "focusin":
        return iu = Bc(
          iu,
          l,
          n,
          u,
          c,
          r
        ), !0;
      case "dragenter":
        return Zu = Bc(
          Zu,
          l,
          n,
          u,
          c,
          r
        ), !0;
      case "mouseover":
        return Ku = Bc(
          Ku,
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
          Bc(
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
          Bc(
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
    var n = Ml(l.target);
    if (n !== null) {
      var u = he(n);
      if (u !== null) {
        if (n = u.tag, n === 13) {
          if (n = Oe(u), n !== null) {
            l.blockedOn = n, Hh(l.priority, function() {
              if (u.tag === 13) {
                var c = Ua();
                c = ll(c);
                var r = Bn(u, c);
                r !== null && _a(r, u, c), Fd(u, c);
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
        Ii = c, u.target.dispatchEvent(c), Ii = null;
      } else
        return n = Ki(u), n !== null && Nm(n), l.blockedOn = u, !1;
      n.shift();
    }
    return !0;
  }
  function Gr(l, n, u) {
    jr(l) && u.delete(n);
  }
  function af() {
    lf = !1, iu !== null && jr(iu) && (iu = null), Zu !== null && jr(Zu) && (Zu = null), Ku !== null && jr(Ku) && (Ku = null), Br.forEach(Gr), Yr.forEach(Gr);
  }
  function eh(l, n) {
    l.blockedOn === n && (l.blockedOn = null, lf || (lf = !0, M.unstable_scheduleCallback(
      M.unstable_NormalPriority,
      af
    )));
  }
  var Yc = null;
  function jm(l) {
    Yc !== l && (Yc = l, M.unstable_scheduleCallback(
      M.unstable_NormalPriority,
      function() {
        Yc === l && (Yc = null);
        for (var n = 0; n < l.length; n += 3) {
          var u = l[n], c = l[n + 1], r = l[n + 2];
          if (typeof c != "function") {
            if (qm(c || u) === null)
              continue;
            break;
          }
          var s = Ki(u);
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
  function cu(l) {
    function n(S) {
      return eh(S, l);
    }
    iu !== null && eh(iu, l), Zu !== null && eh(Zu, l), Ku !== null && eh(Ku, l), Br.forEach(n), Yr.forEach(n);
    for (var u = 0; u < Bi.length; u++) {
      var c = Bi[u];
      c.blockedOn === l && (c.blockedOn = null);
    }
    for (; 0 < Bi.length && (u = Bi[0], u.blockedOn === null); )
      Ym(u), u.blockedOn === null && Bi.shift();
    if (u = (l.ownerDocument || l).$$reactFormReplay, u != null)
      for (c = 0; c < u.length; c += 3) {
        var r = u[c], s = u[c + 1], y = r[kl] || null;
        if (typeof s == "function")
          y || jm(u);
        else if (y) {
          var p = null;
          if (s && s.hasAttribute("formAction")) {
            if (r = s, y = s[kl] || null)
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
      xm(l.current, 2, null, l, null, null), _c(), n[ao] = null;
    }
  };
  function th(l) {
    this._internalRoot = l;
  }
  th.prototype.unstable_scheduleHydration = function(l) {
    if (l) {
      var n = fs();
      l = { blockedOn: null, target: l, priority: n };
      for (var u = 0; u < Bi.length && n !== 0 && n < Bi[u].priority; u++) ;
      Bi.splice(u, 0, l), u === 0 && Ym(l);
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
  W.findDOMNode = function(l) {
    var n = l._reactInternals;
    if (n === void 0)
      throw typeof l.render == "function" ? Error(_(188)) : (l = Object.keys(l).join(","), Error(_(268, l)));
    return l = N(n), l = l !== null ? V(l) : null, l = l === null ? null : l.stateNode, l;
  };
  var ea = {
    bundleType: 0,
    version: "19.1.1",
    rendererPackageName: "react-dom",
    currentDispatcherRef: O,
    reconcilerVersion: "19.1.1"
  };
  if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u") {
    var Lr = __REACT_DEVTOOLS_GLOBAL_HOOK__;
    if (!Lr.isDisabled && Lr.supportsFiber)
      try {
        Pu = Lr.inject(
          ea
        ), Ol = Lr;
      } catch {
      }
  }
  return Rp.createRoot = function(l, n) {
    if (!ie(l)) throw Error(_(299));
    var u = !1, c = "", r = Co, s = Yy, y = dr, p = null;
    return n != null && (n.unstable_strictMode === !0 && (u = !0), n.identifierPrefix !== void 0 && (c = n.identifierPrefix), n.onUncaughtError !== void 0 && (r = n.onUncaughtError), n.onCaughtError !== void 0 && (s = n.onCaughtError), n.onRecoverableError !== void 0 && (y = n.onRecoverableError), n.unstable_transitionCallbacks !== void 0 && (p = n.unstable_transitionCallbacks)), n = _m(
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
    ), l[ao] = n.current, Em(l), new Gm(n);
  }, Rp.hydrateRoot = function(l, n, u) {
    if (!ie(l)) throw Error(_(299));
    var c = !1, r = "", s = Co, y = Yy, p = dr, S = null, x = null;
    return u != null && (u.unstable_strictMode === !0 && (c = !0), u.identifierPrefix !== void 0 && (r = u.identifierPrefix), u.onUncaughtError !== void 0 && (s = u.onUncaughtError), u.onCaughtError !== void 0 && (y = u.onCaughtError), u.onRecoverableError !== void 0 && (p = u.onRecoverableError), u.unstable_transitionCallbacks !== void 0 && (S = u.unstable_transitionCallbacks), u.formState !== void 0 && (x = u.formState)), n = _m(
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
      x
    ), n.context = Cm(null), u = n.current, c = Ua(), c = ll(c), r = ra(c), r.callback = null, Xn(u, r, c), u = c, n.current.lanes = u, we(n, u), pa(n), l[ao] = n.current, Em(l), new th(n);
  }, Rp.version = "19.1.1", Rp;
}
var Op = {}, tS;
function UT() {
  return tS || (tS = 1, It.env.NODE_ENV !== "production" && function() {
    function M(e, t) {
      for (e = e.memoizedState; e !== null && 0 < t; )
        e = e.next, t--;
      return e;
    }
    function F(e, t, a, i) {
      if (a >= t.length) return i;
      var o = t[a], f = qe(e) ? e.slice() : ke({}, e);
      return f[o] = F(e[o], t, a + 1, i), f;
    }
    function re(e, t, a) {
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
    function ie(e, t, a) {
      var i = t[a], o = qe(e) ? e.slice() : ke({}, e);
      return a + 1 === t.length ? (qe(o) ? o.splice(i, 1) : delete o[i], o) : (o[i] = ie(e[i], t, a + 1), o);
    }
    function he() {
      return !1;
    }
    function Oe() {
      return null;
    }
    function Se() {
    }
    function N() {
      console.error(
        "Do not call Hooks inside useEffect(...), useMemo(...), or other built-in Hooks. You can only call Hooks at the top level of your React function. For more information, see https://react.dev/link/rules-of-hooks"
      );
    }
    function V() {
      console.error(
        "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
      );
    }
    function le() {
    }
    function k(e) {
      var t = [];
      return e.forEach(function(a) {
        t.push(a);
      }), t.sort().join(", ");
    }
    function U(e, t, a, i) {
      return new qf(e, t, a, i);
    }
    function ae(e, t) {
      e.context === nf && (Et(e.current, 2, t, e, null, null), Rc());
    }
    function Ye(e, t) {
      if (fu !== null) {
        var a = t.staleFamilies;
        t = t.updatedFamilies, xo(), wf(
          e.current,
          t,
          a
        ), Rc();
      }
    }
    function Mt(e) {
      fu = e;
    }
    function $e(e) {
      return !(!e || e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11);
    }
    function tt(e) {
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
    function Pt(e) {
      if (e.tag === 13) {
        var t = e.memoizedState;
        if (t === null && (e = e.alternate, e !== null && (t = e.memoizedState)), t !== null) return t.dehydrated;
      }
      return null;
    }
    function Me(e) {
      if (tt(e) !== e)
        throw Error("Unable to find node on an unmounted component.");
    }
    function lt(e) {
      var t = e.alternate;
      if (!t) {
        if (t = tt(e), t === null)
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
            if (f === a) return Me(o), e;
            if (f === i) return Me(o), t;
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
    function De(e) {
      var t = e.tag;
      if (t === 5 || t === 26 || t === 27 || t === 6) return e;
      for (e = e.child; e !== null; ) {
        if (t = De(e), t !== null) return t;
        e = e.sibling;
      }
      return null;
    }
    function bt(e) {
      return e === null || typeof e != "object" ? null : (e = Rm && e[Rm] || e["@@iterator"], typeof e == "function" ? e : null);
    }
    function Ge(e) {
      if (e == null) return null;
      if (typeof e == "function")
        return e.$$typeof === Ld ? null : e.displayName || e.name || null;
      if (typeof e == "string") return e;
      switch (e) {
        case Xe:
          return "Fragment";
        case Ko:
          return "Profiler";
        case Zo:
          return "StrictMode";
        case Jo:
          return "Suspense";
        case Ci:
          return "SuspenseList";
        case Am:
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
          case Gd:
            return (e._context.displayName || "Context") + ".Consumer";
          case Gu:
            var t = e.render;
            return e = e.displayName, e || (e = t.displayName || t.name || "", e = e !== "" ? "ForwardRef(" + e + ")" : "ForwardRef"), e;
          case Ur:
            return t = e.displayName || null, t !== null ? t : Ge(e.type) || "Memo";
          case Ca:
            t = e._payload, e = e._init;
            try {
              return Ge(e(t));
            } catch {
            }
        }
      return null;
    }
    function Tt(e) {
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
          return Ge(t);
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
            return de(e.return);
      }
      return null;
    }
    function Rt(e) {
      return { current: e };
    }
    function Te(e, t) {
      0 > Pa ? console.error("Unexpected pop.") : (t !== Cr[Pa] && console.error("Unexpected Fiber popped."), e.current = _r[Pa], _r[Pa] = null, Cr[Pa] = null, Pa--);
    }
    function Ce(e, t, a) {
      Pa++, _r[Pa] = e.current, Cr[Pa] = a, e.current = t;
    }
    function _t(e) {
      return e === null && console.error(
        "Expected host context to exist. This error is likely caused by a bug in React. Please file an issue."
      ), e;
    }
    function Gt(e, t) {
      Ce(nu, t, e), Ce(ko, e, e), Ce(Lu, null, e);
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
                t = Uh;
                break;
              case "math":
                t = rg;
                break;
              default:
                t = kc;
            }
      }
      a = a.toLowerCase(), a = Yh(null, a), a = {
        context: t,
        ancestorInfo: a
      }, Te(Lu, e), Ce(Lu, a, e);
    }
    function pt(e) {
      Te(Lu, e), Te(ko, e), Te(nu, e);
    }
    function O() {
      return _t(Lu.current);
    }
    function W(e) {
      e.memoizedState !== null && Ce($o, e, e);
      var t = _t(Lu.current), a = e.type, i = ya(t.context, a);
      a = Yh(t.ancestorInfo, a), i = { context: i, ancestorInfo: a }, t !== i && (Ce(ko, e, e), Ce(Lu, i, e));
    }
    function P(e) {
      ko.current === e && (Te(Lu, e), Te(ko, e)), $o.current === e && (Te($o, e), bp._currentValue = us);
    }
    function be(e) {
      return typeof Symbol == "function" && Symbol.toStringTag && e[Symbol.toStringTag] || e.constructor.name || "Object";
    }
    function g(e) {
      try {
        return q(e), !1;
      } catch {
        return !0;
      }
    }
    function q(e) {
      return "" + e;
    }
    function K(e, t) {
      if (g(e))
        return console.error(
          "The provided `%s` attribute is an unsupported type %s. This value must be coerced to a string before using it here.",
          t,
          be(e)
        ), q(e);
    }
    function I(e, t) {
      if (g(e))
        return console.error(
          "The provided `%s` CSS property is an unsupported type %s. This value must be coerced to a string before using it here.",
          t,
          be(e)
        ), q(e);
    }
    function ce(e) {
      if (g(e))
        return console.error(
          "Form field values (value, checked, defaultValue, or defaultChecked props) must be strings, not %s. This value must be coerced to a string before using it here.",
          be(e)
        ), q(e);
    }
    function ze(e) {
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
      if (typeof Gg == "function" && Tn(e), wl && typeof wl.setStrictMode == "function")
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
    function Ne() {
      fe !== null && typeof fe.markCommitStopped == "function" && fe.markCommitStopped();
    }
    function wt(e) {
      fe !== null && typeof fe.markComponentRenderStarted == "function" && fe.markComponentRenderStarted(e);
    }
    function na() {
      fe !== null && typeof fe.markComponentRenderStopped == "function" && fe.markComponentRenderStopped();
    }
    function zn(e) {
      fe !== null && typeof fe.markRenderStarted == "function" && fe.markRenderStarted(e);
    }
    function Zi() {
      fe !== null && typeof fe.markRenderStopped == "function" && fe.markRenderStopped();
    }
    function Mn(e, t) {
      fe !== null && typeof fe.markStateUpdateScheduled == "function" && fe.markStateUpdateScheduled(e, t);
    }
    function Pc(e) {
      return e >>>= 0, e === 0 ? 32 : 31 - (Qd(e) / Xu | 0) | 0;
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
    function eo(e) {
      for (var t = [], a = 0; 31 > a; a++) t.push(e);
      return t;
    }
    function bu(e, t) {
      e.pendingLanes |= t, t !== 268435456 && (e.suspendedLanes = 0, e.pingedLanes = 0, e.warmLanes = 0);
    }
    function os(e, t, a, i, o, f) {
      var d = e.pendingLanes;
      e.pendingLanes = a, e.suspendedLanes = 0, e.pingedLanes = 0, e.warmLanes = 0, e.expiredLanes &= a, e.entangledLanes &= a, e.errorRecoveryDisabledLanes &= a, e.shellSuspendCounter = 0;
      var h = e.entanglements, v = e.expirationTimes, b = e.hiddenUpdates;
      for (a = d & ~a; 0 < a; ) {
        var B = 31 - Ql(a), L = 1 << B;
        h[B] = 0, v[B] = -1;
        var H = b[B];
        if (H !== null)
          for (b[B] = null, B = 0; B < H.length; B++) {
            var X = H[B];
            X !== null && (X.lane &= -536870913);
          }
        a &= ~L;
      }
      i !== 0 && Tf(e, i, 0), f !== 0 && o === 0 && e.tag !== 0 && (e.suspendedLanes |= f & ~(d & ~t));
    }
    function Tf(e, t, a) {
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
    function Ya(e, t, a) {
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
      return e &= -e, ql < e ? En < e ? (e & 134217727) !== 0 ? Qu : Jd : En : ql;
    }
    function Ef() {
      var e = xe.p;
      return e !== 0 ? e : (e = window.event, e === void 0 ? Qu : Yd(e.type));
    }
    function lo(e, t) {
      var a = xe.p;
      try {
        return xe.p = e, t();
      } finally {
        xe.p = a;
      }
    }
    function nn(e) {
      delete e[Zl], delete e[ga], delete e[Om], delete e[Ov], delete e[Fo];
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
    function te(e, t) {
      ue(e, t), ue(e + "Capture", t);
    }
    function ue(e, t) {
      en[e] && console.error(
        "EventRegistry: More than one plugin attempted to publish the same registration name, `%s`.",
        e
      ), en[e] = t;
      var a = e.toLowerCase();
      for (wc[a] = e, e === "onDoubleClick" && (wc.ondblclick = e), e = 0; e < t.length; e++)
        Dv.add(t[e]);
    }
    function ve(e, t) {
      zv[t.type] || t.onChange || t.onInput || t.readOnly || t.disabled || t.value == null || console.error(
        e === "select" ? "You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set `onChange`." : "You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set either `onChange` or `readOnly`."
      ), t.onChange || t.readOnly || t.disabled || t.checked == null || console.error(
        "You provided a `checked` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultChecked`. Otherwise, set either `onChange` or `readOnly`."
      );
    }
    function we(e) {
      return Vu.call(Wd, e) ? !0 : Vu.call($d, e) ? !1 : kd.test(e) ? Wd[e] = !0 : ($d[e] = !0, console.error("Invalid attribute name: `%s`", e), !1);
    }
    function Le(e, t, a) {
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
        return e = e.getAttribute(t), e === "" && a === !0 ? !0 : (K(a, t), e === "" + a ? a : e);
      }
    }
    function ct(e, t, a) {
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
          K(a, t), e.setAttribute(t, "" + a);
        }
    }
    function je(e, t, a) {
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
        K(a, t), e.setAttribute(t, "" + a);
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
        K(i, a), e.setAttributeNS(t, a, "" + i);
      }
    }
    function cn() {
    }
    function fs() {
      if (qi === 0) {
        Dm = console.log, zm = console.info, Mv = console.warn, Mm = console.error, ef = console.group, Uv = console.groupCollapsed, _v = console.groupEnd;
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
    function Hh() {
      if (qi--, qi === 0) {
        var e = { configurable: !0, enumerable: !0, writable: !0 };
        Object.defineProperties(console, {
          log: ke({}, e, { value: Dm }),
          info: ke({}, e, { value: zm }),
          warn: ke({}, e, { value: Mv }),
          error: ke({}, e, { value: Mm }),
          group: ke({}, e, { value: ef }),
          groupCollapsed: ke({}, e, { value: Uv }),
          groupEnd: ke({}, e, { value: _v })
        });
      }
      0 > qi && console.error(
        "disabledDepth fell below zero. This is a bug in React. Please file an issue."
      );
    }
    function cl(e) {
      if (Um === void 0)
        try {
          throw Error();
        } catch (a) {
          var t = a.stack.trim().match(/\n( *(at )?)/);
          Um = t && t[1] || "", Nr = -1 < a.stack.indexOf(`
    at`) ? " (<anonymous>)" : -1 < a.stack.indexOf("@") ? "@unknown:0:0" : "";
        }
      return `
` + Um + e + Nr;
    }
    function vl(e, t) {
      if (!e || tf) return "";
      var a = wr.get(e);
      if (a !== void 0) return a;
      tf = !0, a = Error.prepareStackTrace, Error.prepareStackTrace = void 0;
      var i = null;
      i = j.H, j.H = null, fs();
      try {
        var o = {
          DetermineComponentFrameRoot: function() {
            try {
              if (t) {
                var H = function() {
                  throw Error();
                };
                if (Object.defineProperty(H.prototype, "props", {
                  set: function() {
                    throw Error();
                  }
                }), typeof Reflect == "object" && Reflect.construct) {
                  try {
                    Reflect.construct(H, []);
                  } catch (me) {
                    var X = me;
                  }
                  Reflect.construct(e, [], H);
                } else {
                  try {
                    H.call();
                  } catch (me) {
                    X = me;
                  }
                  e.call(H.prototype);
                }
              } else {
                try {
                  throw Error();
                } catch (me) {
                  X = me;
                }
                (H = e()) && typeof H.catch == "function" && H.catch(function() {
                });
              }
            } catch (me) {
              if (me && X && typeof me.stack == "string")
                return [me.stack, X.stack];
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
        tf = !1, j.H = i, Hh(), Error.prepareStackTrace = a;
      }
      return b = (b = e ? e.displayName || e.name : "") ? cl(b) : "", typeof e == "function" && wr.set(e, b), b;
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
    function rs(e) {
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
    function zp(e) {
      return (e = e ? e.displayName || e.name : "") ? cl(e) : "";
    }
    function ss() {
      if (xa === null) return null;
      var e = xa._debugOwner;
      return e != null ? Tt(e) : null;
    }
    function Mp() {
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
      Af(e);
      try {
        return e !== null && e._debugTask ? e._debugTask.run(
          t.bind(null, a, i, o, f, d)
        ) : t(a, i, o, f, d);
      } finally {
        Af(h);
      }
      throw Error(
        "runWithFiberInDEV should never be called in production. This is a bug in React."
      );
    }
    function Af(e) {
      j.getCurrentStack = e === null ? null : Mp, ba = !1, xa = e;
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
    function Rf(e) {
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
    function Su(e) {
      e._valueTracker || (e._valueTracker = Rf(e));
    }
    function ol(e) {
      if (!e) return !1;
      var t = e._valueTracker;
      if (!t) return !0;
      var a = t.getValue(), i = "";
      return e && (i = Ki(e) ? e.checked ? "true" : "false" : e.value), e = i, e !== a ? (t.setValue(e), !0) : !1;
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
    function ei(e, t) {
      t.checked === void 0 || t.defaultChecked === void 0 || Cm || (console.error(
        "%s contains an input of type %s with both checked and defaultChecked props. Input elements must be either controlled or uncontrolled (specify either the checked prop, or the defaultChecked prop, but not both). Decide between using a controlled or uncontrolled input element and remove one of these props. More info: https://react.dev/link/controlled-components",
        ss() || "A component",
        t.type
      ), Cm = !0), t.value === void 0 || t.defaultValue === void 0 || _m || (console.error(
        "%s contains an input of type %s with both value and defaultValue props. Input elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled input element and remove one of these props. More info: https://react.dev/link/controlled-components",
        ss() || "A component",
        t.type
      ), _m = !0);
    }
    function ti(e, t, a, i, o, f, d, h) {
      e.name = "", d != null && typeof d != "function" && typeof d != "symbol" && typeof d != "boolean" ? (K(d, "type"), e.type = d) : e.removeAttribute("type"), t != null ? d === "number" ? (t === 0 && e.value === "" || e.value != t) && (e.value = "" + Ml(t)) : e.value !== "" + Ml(t) && (e.value = "" + Ml(t)) : d !== "submit" && d !== "reset" || e.removeAttribute("value"), t != null ? ds(e, d, Ml(t)) : a != null ? ds(e, d, Ml(a)) : i != null && e.removeAttribute("value"), o == null && f != null && (e.defaultChecked = !!f), o != null && (e.checked = o && typeof o != "function" && typeof o != "symbol"), h != null && typeof h != "function" && typeof h != "symbol" && typeof h != "boolean" ? (K(h, "name"), e.name = "" + Ml(h)) : e.removeAttribute("name");
    }
    function Up(e, t, a, i, o, f, d, h) {
      if (f != null && typeof f != "function" && typeof f != "symbol" && typeof f != "boolean" && (K(f, "type"), e.type = f), t != null || a != null) {
        if (!(f !== "submit" && f !== "reset" || t != null))
          return;
        a = a != null ? "" + Ml(a) : "", t = t != null ? "" + Ml(t) : a, h || t === e.value || (e.value = t), e.defaultValue = t;
      }
      i = i ?? o, i = typeof i != "function" && typeof i != "symbol" && !!i, e.checked = h ? e.checked : !!i, e.defaultChecked = !!i, d != null && typeof d != "function" && typeof d != "symbol" && typeof d != "boolean" && (K(d, "name"), e.name = d);
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
    function _p() {
      var e = ss();
      return e ? `

Check the render method of \`` + e + "`." : "";
    }
    function Tu(e, t, a, i) {
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
    function Df(e, t) {
      for (e = 0; e < qr.length; e++) {
        var a = qr[e];
        if (t[a] != null) {
          var i = qe(t[a]);
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
      t.value === void 0 || t.defaultValue === void 0 || Nm || (console.error(
        "Select elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled select element and remove one of these props. More info: https://react.dev/link/controlled-components"
      ), Nm = !0);
    }
    function _n(e, t) {
      t.value === void 0 || t.defaultValue === void 0 || Cv || (console.error(
        "%s contains a textarea with both value and defaultValue props. Textarea elements must be either controlled or uncontrolled (specify either the value prop, or the defaultValue prop, but not both). Decide between using a controlled or uncontrolled textarea and remove one of these props. More info: https://react.dev/link/controlled-components",
        ss() || "A component"
      ), Cv = !0), t.children != null && t.value == null && console.error(
        "Use the `defaultValue` or `value` props instead of setting children on <textarea>."
      );
    }
    function hs(e, t, a) {
      if (t != null && (t = "" + Ml(t), t !== e.value && (e.value = t), a == null)) {
        e.defaultValue !== t && (e.defaultValue = t);
        return;
      }
      e.defaultValue = a != null ? "" + Ml(a) : "";
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
    function Gl(e, t) {
      return xv.test(e) ? (e = JSON.stringify(e), e.length > t - 2 ? 8 > t ? '{"..."}' : "{" + e.slice(0, t - 7) + '..."}' : "{" + e + "}") : e.length > t ? 5 > t ? '{"..."}' : e.slice(0, t - 3) + "..." : e;
    }
    function zf(e, t, a) {
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
    function Bh(e) {
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
          if (qe(e)) return "[...]";
          if (e.$$typeof === _i)
            return (t = Ge(e.type)) ? "<" + t + ">" : "<...>";
          var a = Bh(e);
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
      return typeof e != "string" || xv.test(e) ? "{" + ai(e, t - 2) + "}" : e.length > t - 2 ? 5 > t ? '"..."' : '"' + e.slice(0, t - 5) + '..."' : '"' + e + '"';
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
    function Ag(e, t, a) {
      var i = "", o = ke({}, t), f;
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
    function ja(e, t, a, i) {
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
              ), typeof b == "object" && b !== null && typeof v == "object" && v !== null && Bh(b) === "Object" && Bh(v) === "Object" && (2 < Object.keys(b).length || 2 < Object.keys(v).length || -1 < B.indexOf("...") || -1 < h.indexOf("...")) ? o += $l(i + 1) + d + `={{
` + Ag(
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
        f.forEach(function(L) {
          if (L !== "children") {
            var H = 120 - 2 * (i + 1) - L.length - 1;
            o += ki(i + 1) + L + "=" + $i(a[L], H) + `
`;
          }
        }), o = o === "" ? $l(i) + "<" + e + `>
` : $l(i) + "<" + e + `
` + o + $l(i) + `>
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
      return $l(t) + "<" + a + `>
`;
    }
    function ms(e, t) {
      var a = Ji(e, t);
      if (a !== e && (e.children.length !== 1 || e.children[0] !== a))
        return $l(t) + `...
` + ms(a, t + 1);
      a = "";
      var i = e.fiber._debugInfo;
      if (i)
        for (var o = 0; o < i.length; o++) {
          var f = i[o].name;
          typeof f == "string" && (a += $l(t) + "<" + f + `>
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
          ) : (i = ja(
            f,
            o,
            e.serverProps,
            t
          ), t++);
      var b = "";
      for (o = e.fiber.child, f = 0; o && f < e.children.length; )
        d = e.children[f], d.fiber === o ? (b += ms(d, t), f++) : b += ys(o, t), o = o.sibling;
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
    function Mf(e) {
      try {
        return `

` + ms(e, 0);
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
      ) ? null : a) ? null : uo(e, t), t = a || t, !t) return !0;
      var i = t.tag;
      if (t = String(!!a) + "|" + e + "|" + i, lf[t]) return !1;
      lf[t] = !0;
      var o = (t = xa) ? Cp(t.return, i) : null, f = t !== null && o !== null ? Wi(o, t, null) : "", d = "<" + e + ">";
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
    function Uf(e, t, a) {
      if (a || jh("#text", t, !1))
        return !0;
      if (a = "#text|" + t, lf[a]) return !1;
      lf[a] = !0;
      var i = (a = xa) ? Cp(a, t) : null;
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
    function Rg(e) {
      return e.replace(Bi, function(t, a) {
        return a.toUpperCase();
      });
    }
    function xp(e, t, a) {
      var i = t.indexOf("--") === 0;
      i || (-1 < t.indexOf("-") ? qc.hasOwnProperty(t) && qc[t] || (qc[t] = !0, console.error(
        "Unsupported style property %s. Did you mean %s?",
        t,
        Rg(t.replace(Yr, "ms-"))
      )) : Br.test(t) ? qc.hasOwnProperty(t) && qc[t] || (qc[t] = !0, console.error(
        "Unsupported vendor-prefixed style property %s. Did you mean %s?",
        t,
        t.charAt(0).toUpperCase() + t.slice(1)
      )) : !Hv.test(a) || Bc.hasOwnProperty(a) && Bc[a] || (Bc[a] = !0, console.error(
        `Style property values shouldn't contain a semicolon. Try "%s: %s" instead.`,
        t,
        a.replace(Hv, "")
      )), typeof a == "number" && (isNaN(a) ? Nv || (Nv = !0, console.error(
        "`NaN` is an invalid value for the `%s` css style property.",
        t
      )) : isFinite(a) || Ym || (Ym = !0, console.error(
        "`Infinity` is an invalid value for the `%s` css style property.",
        t
      )))), a == null || typeof a == "boolean" || a === "" ? i ? e.setProperty(t, "") : t === "float" ? e.cssFloat = "" : e[t] = "" : i ? e.setProperty(t, a) : typeof a != "number" || a === 0 || jr.has(t) ? t === "float" ? e.cssFloat = a : (I(a, t), e[t] = ("" + a).trim()) : e[t] = a + "px";
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
                for (var f = iu[o] || [o], d = 0; d < f.length; d++)
                  i[f[d]] = o;
          }
          for (var h in t)
            if (t.hasOwnProperty(h) && (!a || a[h] !== t[h]))
              for (o = iu[h] || [h], f = 0; f < o.length; f++)
                i[o[f]] = h;
          h = {};
          for (var v in t)
            for (o = iu[v] || [v], f = 0; f < o.length; f++)
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
        for (var L in a)
          !a.hasOwnProperty(L) || t != null && t.hasOwnProperty(L) || (L.indexOf("--") === 0 ? e.setProperty(L, "") : L === "float" ? e.cssFloat = "" : e[L] = "");
        for (var H in t)
          b = t[H], t.hasOwnProperty(H) && a[H] !== b && xp(e, H, b);
      } else
        for (i in t)
          t.hasOwnProperty(i) && xp(e, i, t[i]);
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
    function vs(e) {
      return eh.get(e) || e;
    }
    function io(e, t) {
      if (Vu.call(cu, t) && cu[t])
        return !0;
      if (th.test(t)) {
        if (e = "aria-" + t.slice(4).toLowerCase(), e = jm.hasOwnProperty(e) ? e : null, e == null)
          return console.error(
            "Invalid ARIA attribute `%s`. ARIA attributes follow the pattern aria-* and must be lowercase.",
            t
          ), cu[t] = !0;
        if (t !== e)
          return console.error(
            "Invalid ARIA attribute `%s`. Did you mean `%s`?",
            t,
            e
          ), cu[t] = !0;
      }
      if (Gm.test(t)) {
        if (e = t.toLowerCase(), e = jm.hasOwnProperty(e) ? e : null, e == null) return cu[t] = !0, !1;
        t !== e && (console.error(
          "Unknown ARIA attribute `%s`. Did you mean `%s`?",
          t,
          e
        ), cu[t] = !0);
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
    function Hp(e, t, a, i) {
      if (Vu.call(ea, t) && ea[t])
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
        if (Lr.test(t))
          return console.error(
            "Unknown event handler property `%s`. It will be ignored.",
            t
          ), ea[t] = !0;
      } else if (Lr.test(t))
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
      if (Yc.hasOwnProperty(o)) {
        if (o = Yc[o], o !== t)
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
    function oo(e) {
      return c.test("" + e) ? "javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')" : e;
    }
    function Pi(e) {
      return e = e.target || e.srcElement || window, e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e;
    }
    function Cn(e) {
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
              for (K(t, "name"), a = a.querySelectorAll(
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
            hs(e, a.value, a.defaultValue);
            break e;
          case "select":
            t = a.value, t != null && Tu(e, !!a.multiple, t, !1);
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
        if (p = !1, (s !== null || y !== null) && (Rc(), s && (t = s, e = y, y = s = null, Cn(t), e)))
          for (t = 0; t < e.length; t++) Cn(e[t]);
      }
    }
    function Eu(e, t) {
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
    function Au() {
      if (Y) return Y;
      var e, t = w, a = t.length, i, o = "value" in $ ? $.value : $.textContent, f = o.length;
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
    function Lh() {
      return !1;
    }
    function Ul(e) {
      function t(a, i, o, f, d) {
        this._reactName = a, this._targetInst = o, this.type = i, this.nativeEvent = f, this.target = d, this.currentTarget = null;
        for (var h in e)
          e.hasOwnProperty(h) && (a = e[h], this[h] = a ? a(f) : f[h]);
        return this.isDefaultPrevented = (f.defaultPrevented != null ? f.defaultPrevented : f.returnValue === !1) ? ec : Lh, this.isPropagationStopped = Lh, this;
      }
      return ke(t.prototype, {
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
    function bs(e) {
      var t = this.nativeEvent;
      return t.getModifierState ? t.getModifierState(e) : (e = pS[e]) ? !!t[e] : !1;
    }
    function Ss() {
      return bs;
    }
    function Wl(e, t) {
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
    function ni(e) {
      return e = e.detail, typeof e == "object" && "data" in e ? e.data : null;
    }
    function Ts(e, t) {
      switch (e) {
        case "compositionend":
          return ni(t);
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
        return e === "compositionend" || !Xg && Wl(e, t) ? (e = Au(), Y = w = $ = null, lh = !1, e) : null;
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
      return t === "input" ? !!_S[e.type] : t === "textarea";
    }
    function Vh(e) {
      if (!S) return !1;
      e = "on" + e;
      var t = e in document;
      return t || (t = document.createElement("div"), t.setAttribute(e, "return;"), t = typeof t[e] == "function"), t;
    }
    function Es(e, t, a, i) {
      s ? y ? y.push(i) : y = [i] : s = i, t = gr(t, "onChange"), 0 < t.length && (a = new Re(
        "onChange",
        "change",
        null,
        a,
        i
      ), e.push({ event: a, listeners: t }));
    }
    function xf(e) {
      In(e, 0);
    }
    function tc(e) {
      var t = un(e);
      if (ol(t)) return e;
    }
    function Xh(e, t) {
      if (e === "change") return t;
    }
    function wp() {
      Xm && (Xm.detachEvent("onpropertychange", qp), Qm = Xm = null);
    }
    function qp(e) {
      if (e.propertyName === "value" && tc(Qm)) {
        var t = [];
        Es(
          t,
          Qm,
          e,
          Pi(e)
        ), gs(xf, t);
      }
    }
    function Og(e, t, a) {
      e === "focusin" ? (wp(), Xm = t, Qm = a, Xm.attachEvent("onpropertychange", qp)) : e === "focusout" && wp();
    }
    function Qh(e) {
      if (e === "selectionchange" || e === "keyup" || e === "keydown")
        return tc(Qm);
    }
    function Dg(e, t) {
      if (e === "click") return tc(t);
    }
    function zg(e, t) {
      if (e === "input" || e === "change")
        return tc(t);
    }
    function Mg(e, t) {
      return e === t && (e !== 0 || 1 / e === 1 / t) || e !== e && t !== t;
    }
    function Hf(e, t) {
      if (Ha(e, t)) return !0;
      if (typeof e != "object" || e === null || typeof t != "object" || t === null)
        return !1;
      var a = Object.keys(e), i = Object.keys(t);
      if (a.length !== i.length) return !1;
      for (i = 0; i < a.length; i++) {
        var o = a[i];
        if (!Vu.call(t, o) || !Ha(e[o], t[o]))
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
      }), Zm && Hf(Zm, i) || (Zm = i, i = gr(Qg, "onSelect"), 0 < i.length && (t = new Re(
        "onSelect",
        "select",
        null,
        t,
        a
      ), e.push({ event: t, listeners: i }), t.target = ah)));
    }
    function Ru(e, t) {
      var a = {};
      return a[e.toLowerCase()] = t.toLowerCase(), a["Webkit" + e] = "webkit" + t, a["Moz" + e] = "moz" + t, a;
    }
    function lc(e) {
      if (Kg[e]) return Kg[e];
      if (!nh[e]) return e;
      var t = nh[e], a;
      for (a in t)
        if (t.hasOwnProperty(a) && a in I0)
          return Kg[e] = t[a];
      return e;
    }
    function on(e, t) {
      a1.set(e, t), te(t, [e]);
    }
    function Ra(e, t) {
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
        var a = ou[t];
        ou[t++] = null;
        var i = ou[t];
        ou[t++] = null;
        var o = ou[t];
        ou[t++] = null;
        var f = ou[t];
        if (ou[t++] = null, i !== null && o !== null) {
          var d = i.pending;
          d === null ? o.next = o : (o.next = d.next, d.next = o), i.pending = o;
        }
        f !== 0 && Lp(a, o, f);
      }
    }
    function As(e, t, a, i) {
      ou[uh++] = e, ou[uh++] = t, ou[uh++] = a, ou[uh++] = i, $g |= i, e.lanes |= i, e = e.alternate, e !== null && (e.lanes |= i);
    }
    function Jh(e, t, a, i) {
      return As(e, t, a, i), Rs(e);
    }
    function ia(e, t) {
      return As(e, null, null, t), Rs(e);
    }
    function Lp(e, t, a) {
      e.lanes |= a;
      var i = e.alternate;
      i !== null && (i.lanes |= a);
      for (var o = !1, f = e.return; f !== null; )
        f.childLanes |= a, i = f.alternate, i !== null && (i.childLanes |= a), f.tag === 22 && (e = f.stateNode, e === null || e._visibility & wv || (o = !0)), e = f, f = f.return;
      return e.tag === 3 ? (f = e.stateNode, o && t !== null && (o = 31 - Ql(a), e = f.hiddenUpdates, i = e[o], i === null ? e[o] = [t] : i.push(t), t.lane = a | 536870912), f) : null;
    }
    function Rs(e) {
      if (hp > PS)
        throw es = hp = 0, yp = O0 = null, Error(
          "Maximum update depth exceeded. This can happen when a component repeatedly calls setState inside componentWillUpdate or componentDidUpdate. React limits the number of nested updates to prevent infinite loops."
        );
      es > eT && (es = 0, yp = null, console.error(
        "Maximum update depth exceeded. This can happen when a component calls setState inside useEffect, but useEffect either doesn't have a dependency array, or one of the dependencies changes on every render."
      )), e.alternate === null && (e.flags & 4098) !== 0 && Sn(e);
      for (var t = e, a = t.return; a !== null; )
        t.alternate === null && (t.flags & 4098) !== 0 && Sn(e), t = a, a = t.return;
      return t.tag === 3 ? t.stateNode : null;
    }
    function ac(e) {
      if (fu === null) return e;
      var t = fu(e);
      return t === void 0 ? e : t.current;
    }
    function kh(e) {
      if (fu === null) return e;
      var t = fu(e);
      return t === void 0 ? e != null && typeof e.render == "function" && (t = ac(e.render), e.render !== t) ? (t = { $$typeof: Gu, render: t }, e.displayName !== void 0 && (t.displayName = e.displayName), t) : e : t.current;
    }
    function Vp(e, t) {
      if (fu === null) return !1;
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
          (o === Gu || o === Ca) && (i = !0);
          break;
        case 14:
        case 15:
          (o === Ur || o === Ca) && (i = !0);
          break;
        default:
          return !1;
      }
      return !!(i && (e = fu(a), e !== void 0 && e === fu(t)));
    }
    function Xp(e) {
      fu !== null && typeof WeakSet == "function" && (ih === null && (ih = /* @__PURE__ */ new WeakSet()), ih.add(e));
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
      if (fu === null)
        throw Error("Expected resolveFamily to be set during hot reload.");
      var b = !1;
      h = !1, v !== null && (v = fu(v), v !== void 0 && (a.has(v) ? h = !0 : t.has(v) && (d === 1 ? h = !0 : b = !0))), ih !== null && (ih.has(e) || i !== null && ih.has(i)) && (h = !0), h && (e._debugNeedsRemount = !0), (h || b) && (i = ia(e, 2), i !== null && Kt(i, e, 2)), o === null || h || wf(
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
    function xn(e, t) {
      var a = e.alternate;
      switch (a === null ? (a = U(
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
        $h(e) && (d = 1), h = ac(h);
      else if (typeof e == "string")
        d = O(), d = Xo(e, a, d) ? 26 : e === "html" || e === "head" || e === "body" ? 27 : 5;
      else
        e: switch (e) {
          case Am:
            return t = U(31, a, t, o), t.elementType = Am, t.lanes = f, t;
          case Xe:
            return ui(
              a.children,
              o,
              f,
              t
            );
          case Zo:
            d = 8, o |= Sa, o |= Ju;
            break;
          case Ko:
            return e = a, i = o, typeof e.id != "string" && console.error(
              'Profiler must specify an "id" of type `string` as a prop. Received the type `%s` instead.',
              typeof e.id
            ), t = U(12, e, t, i | ta), t.elementType = Ko, t.lanes = f, t.stateNode = { effectDuration: 0, passiveEffectDuration: 0 }, t;
          case Jo:
            return t = U(13, a, t, o), t.elementType = Jo, t.lanes = f, t;
          case Ci:
            return t = U(19, a, t, o), t.elementType = Ci, t.lanes = f, t;
          default:
            if (typeof e == "object" && e !== null)
              switch (e.$$typeof) {
                case Em:
                case Ia:
                  d = 10;
                  break e;
                case Gd:
                  d = 9;
                  break e;
                case Gu:
                  d = 11, h = kh(h);
                  break e;
                case Ur:
                  d = 14;
                  break e;
                case Ca:
                  d = 16, h = null;
                  break e;
              }
            h = "", (e === void 0 || typeof e == "object" && e !== null && Object.keys(e).length === 0) && (h += " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports."), e === null ? a = "null" : qe(e) ? a = "array" : e !== void 0 && e.$$typeof === _i ? (a = "<" + (Ge(e.type) || "Unknown") + " />", h = " Did you accidentally export a JSX literal instead of a component?") : a = typeof e, (d = i ? Tt(i) : null) && (h += `

Check the render method of \`` + d + "`."), d = 29, a = Error(
              "Element type is invalid: expected a string (for built-in components) or a class/function (for composite components) but got: " + (a + "." + h)
            ), h = null;
        }
      return t = U(d, a, t, o), t.elementType = e, t.type = h, t.lanes = f, t._debugOwner = i, t;
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
    function ui(e, t, a, i) {
      return e = U(7, e, i, t), e.lanes = a, e;
    }
    function ii(e, t, a) {
      return e = U(6, e, null, t), e.lanes = a, e;
    }
    function Fh(e, t, a) {
      return t = U(
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
      fn(), ch[oh++] = Bv, ch[oh++] = qv, qv = e, Bv = t;
    }
    function Qp(e, t, a) {
      fn(), ru[su++] = Gc, ru[su++] = Lc, ru[su++] = Vr, Vr = e;
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
    function Ds(e) {
      fn(), e.return !== null && (nc(e, 1), Qp(e, 1, 0));
    }
    function zs(e) {
      for (; e === qv; )
        qv = ch[--oh], ch[oh] = null, Bv = ch[--oh], ch[oh] = null;
      for (; e === Vr; )
        Vr = ru[--su], ru[su] = null, Lc = ru[--su], ru[su] = null, Gc = ru[--su], ru[su] = null;
    }
    function fn() {
      mt || console.error(
        "Expected to be hydrating. This is a bug in React. Please file an issue."
      );
    }
    function rn(e, t) {
      if (e.return === null) {
        if (du === null)
          du = {
            fiber: e,
            children: [],
            serverProps: void 0,
            serverTail: [],
            distanceFromLeaf: t
          };
        else {
          if (du.fiber !== e)
            throw Error(
              "Saw multiple hydration diff roots in a pass. This is a bug in React."
            );
          du.distanceFromLeaf > t && (du.distanceFromLeaf = t);
        }
        return du;
      }
      var a = rn(
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
      Vc || (e = rn(e, 0), e.serverProps = null, t !== null && (t = _d(t), e.serverTail.push(t)));
    }
    function Hn(e) {
      var t = "", a = du;
      throw a !== null && (du = null, t = Mf(a)), ro(
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
      ), Wg;
    }
    function Ph(e) {
      var t = e.stateNode, a = e.type, i = e.memoizedProps;
      switch (t[Zl] = e, t[ga] = i, Pn(a, i), a) {
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
          ve("input", i), et("invalid", t), ei(t, i), Up(
            t,
            i.value,
            i.defaultValue,
            i.checked,
            i.defaultChecked,
            i.type,
            i.name,
            !0
          ), Su(t);
          break;
        case "option":
          Nh(t, i);
          break;
        case "select":
          ve("select", i), et("invalid", t), Df(t, i);
          break;
        case "textarea":
          ve("textarea", i), et("invalid", t), _n(t, i), wh(
            t,
            i.value,
            i.defaultValue,
            i.children
          ), Su(t);
      }
      a = i.children, typeof a != "string" && typeof a != "number" && typeof a != "bigint" || t.textContent === "" + a || i.suppressHydrationWarning === !0 || tm(t.textContent, a) ? (i.popover != null && (et("beforetoggle", t), et("toggle", t)), i.onScroll != null && et("scroll", t), i.onScrollEnd != null && et("scrollend", t), i.onClick != null && (t.onclick = qu), t = !0) : t = !1, t || Hn(e);
    }
    function ey(e) {
      for (Na = e.return; Na; )
        switch (Na.tag) {
          case 5:
          case 13:
            ji = !1;
            return;
          case 27:
          case 3:
            ji = !0;
            return;
          default:
            Na = Na.return;
        }
    }
    function uc(e) {
      if (e !== Na) return !1;
      if (!mt)
        return ey(e), mt = !0, !1;
      var t = e.tag, a;
      if ((a = t !== 3 && t !== 27) && ((a = t === 5) && (a = e.type, a = !(a !== "form" && a !== "button") || eu(e.type, e.memoizedProps)), a = !a), a && nl) {
        for (a = nl; a; ) {
          var i = rn(e, 0), o = _d(a);
          i.serverTail.push(o), a = o.type === "Suspense" ? fm(a) : Nl(a.nextSibling);
        }
        Hn(e);
      }
      if (ey(e), t === 13) {
        if (e = e.memoizedState, e = e !== null ? e.dehydrated : null, !e)
          throw Error(
            "Expected to have a hydrated suspense instance. This error is likely caused by a bug in React. Please file an issue."
          );
        nl = fm(e);
      } else
        t === 27 ? (t = nl, tu(e.type) ? (e = B0, B0 = null, nl = e) : nl = t) : nl = Na ? Nl(e.stateNode.nextSibling) : null;
      return !0;
    }
    function ic() {
      nl = Na = null, Vc = mt = !1;
    }
    function ty() {
      var e = Xr;
      return e !== null && (Ba === null ? Ba = e : Ba.push.apply(
        Ba,
        e
      ), Xr = null), e;
    }
    function ro(e) {
      Xr === null ? Xr = [e] : Xr.push(e);
    }
    function ly() {
      var e = du;
      if (e !== null) {
        du = null;
        for (var t = Mf(e); 0 < e.children.length; )
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
    function Ms() {
      fh = Yv = null, rh = !1;
    }
    function ci(e, t, a) {
      Ce(Fg, t._currentValue, e), t._currentValue = a, Ce(Ig, t._currentRenderer, e), t._currentRenderer !== void 0 && t._currentRenderer !== null && t._currentRenderer !== f1 && console.error(
        "Detected multiple renderers concurrently rendering the same context provider. This is currently unsupported."
      ), t._currentRenderer = f1;
    }
    function Ou(e, t) {
      e._currentValue = Fg.current;
      var a = Ig.current;
      Te(Ig, t), e._currentRenderer = a, Te(Fg, t);
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
    function _l(e, t, a, i) {
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
      Yv = e, fh = null, e = e.dependencies, e !== null && (e.firstContext = null);
    }
    function xt(e) {
      return rh && console.error(
        "Context can only be read while React is rendering. In classes, you can read it in the render method or getDerivedStateFromProps. In function components, you can read it directly in the function body, but not inside Hooks like useReducer() or useMemo()."
      ), uy(Yv, e);
    }
    function Yf(e, t) {
      return Yv === null && fi(e), uy(e, t);
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
    function cc(e) {
      e.controller.signal.aborted && console.warn(
        "A cache instance was retained after it was already freed. This likely indicates a bug in React."
      ), e.refCount++;
    }
    function Nn(e) {
      e.refCount--, 0 > e.refCount && console.warn(
        "A cache instance was released after it was already freed. This likely indicates a bug in React."
      ), e.refCount === 0 && jS(GS, function() {
        e.controller.abort();
      });
    }
    function sn() {
      var e = Qr;
      return Qr = 0, e;
    }
    function ri(e) {
      var t = Qr;
      return Qr = e, t;
    }
    function oc(e) {
      var t = Qr;
      return Qr += e, t;
    }
    function Us(e) {
      tn = sh(), 0 > e.actualStartTime && (e.actualStartTime = tn);
    }
    function Du(e) {
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
        tn = -1, Qr += e;
      }
    }
    function dn() {
      tn = sh();
    }
    function wn(e) {
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
      return e !== null ? e : Ht.pooledCache;
    }
    function _s(e, t) {
      t === null ? Ce(Kr, Kr.current, e) : Ce(Kr, t.pool, e);
    }
    function Jp() {
      var e = cy();
      return e === null ? null : { parent: Bl._currentValue, pool: e };
    }
    function oy() {
      return { didWarnAboutUncachedPromise: !1, thenables: [] };
    }
    function fy(e) {
      return e = e.status, e === "fulfilled" || e === "rejected";
    }
    function so() {
    }
    function La(e, t, a) {
      j.actQueue !== null && (j.didUsePromise = !0);
      var i = e.thenables;
      switch (a = i[a], a === void 0 ? i.push(t) : a !== t && (e.didWarnAboutUncachedPromise || (e.didWarnAboutUncachedPromise = !0, console.error(
        "A component was suspended by an uncached promise. Creating promises inside a Client Component or hook is not yet supported, except via a Suspense-compatible library or framework."
      )), t.then(so, so), t = a), t.status) {
        case "fulfilled":
          return t.value;
        case "rejected":
          throw e = t.reason, Oa(e), e;
        default:
          if (typeof t.status == "string")
            t.then(so, so);
          else {
            if (e = Ht, e !== null && 100 < e.shellSuspendCounter)
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
    function Oa(e) {
      if (e === Pm || e === Xv)
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
    function si(e, t) {
      e = e.updateQueue, t.updateQueue === e && (t.updateQueue = {
        baseState: e.baseState,
        firstBaseUpdate: e.firstBaseUpdate,
        lastBaseUpdate: e.lastBaseUpdate,
        shared: e.shared,
        callbacks: null
      });
    }
    function qn(e) {
      return {
        lane: e,
        tag: y1,
        payload: null,
        callback: null,
        next: null
      };
    }
    function hn(e, t, a) {
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
      return (At & qa) !== An ? (o = i.pending, o === null ? t.next = t : (t.next = o.next, o.next = t), i.pending = t, t = Rs(e), Lp(e, null, a), t) : (As(e, i, t, a), Rs(e));
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
    function Bn() {
      if (a0) {
        var e = dh;
        if (e !== null) throw e;
      }
    }
    function yo(e, t, a, i) {
      a0 = !1;
      var o = e.updateQueue;
      uf = !1, l0 = o.shared;
      var f = o.firstBaseUpdate, d = o.lastBaseUpdate, h = o.shared.pending;
      if (h !== null) {
        o.shared.pending = null;
        var v = h, b = v.next;
        v.next = null, d === null ? f = b : d.next = b, d = v;
        var B = e.alternate;
        B !== null && (B = B.updateQueue, h = B.lastBaseUpdate, h !== d && (h === null ? B.firstBaseUpdate = b : h.next = b, B.lastBaseUpdate = v));
      }
      if (f !== null) {
        var L = o.baseState;
        d = 0, B = b = v = null, h = f;
        do {
          var H = h.lane & -536870913, X = H !== h.lane;
          if (X ? (it & H) === H : (i & H) === H) {
            H !== 0 && H === Zr && (a0 = !0), B !== null && (B = B.next = {
              lane: 0,
              tag: h.tag,
              payload: h.payload,
              callback: null,
              next: null
            });
            e: {
              H = e;
              var me = h, He = t, Nt = a;
              switch (me.tag) {
                case m1:
                  if (me = me.payload, typeof me == "function") {
                    rh = !0;
                    var ft = me.call(
                      Nt,
                      L,
                      He
                    );
                    if (H.mode & Sa) {
                      oe(!0);
                      try {
                        me.call(Nt, L, He);
                      } finally {
                        oe(!1);
                      }
                    }
                    rh = !1, L = ft;
                    break e;
                  }
                  L = me;
                  break e;
                case t0:
                  H.flags = H.flags & -65537 | 128;
                case y1:
                  if (ft = me.payload, typeof ft == "function") {
                    if (rh = !0, me = ft.call(
                      Nt,
                      L,
                      He
                    ), H.mode & Sa) {
                      oe(!0);
                      try {
                        ft.call(Nt, L, He);
                      } finally {
                        oe(!1);
                      }
                    }
                    rh = !1;
                  } else me = ft;
                  if (me == null) break e;
                  L = ke({}, L, me);
                  break e;
                case p1:
                  uf = !0;
              }
            }
            H = h.callback, H !== null && (e.flags |= 64, X && (e.flags |= 8192), X = o.callbacks, X === null ? o.callbacks = [H] : X.push(H));
          } else
            X = {
              lane: H,
              tag: h.tag,
              payload: h.payload,
              callback: h.callback,
              next: null
            }, B === null ? (b = B = X, v = L) : B = B.next = X, d |= H;
          if (h = h.next, h === null) {
            if (h = o.shared.pending, h === null)
              break;
            X = h, h = X.next, X.next = null, o.lastBaseUpdate = X, o.shared.pending = null;
          }
        } while (!0);
        B === null && (v = L), o.baseState = v, o.firstBaseUpdate = b, o.lastBaseUpdate = B, f === null && (o.shared.lanes = 0), rf |= d, e.lanes = d, e.memoizedState = L;
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
    function mo(e, t) {
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
    function oa(e, t) {
      var a = Vi;
      Ce(Zv, a, e), Ce(hh, t, e), Vi = a | t.baseLanes;
    }
    function Lf(e) {
      Ce(Zv, Vi, e), Ce(
        hh,
        hh.current,
        e
      );
    }
    function yn(e) {
      Vi = Zv.current, Te(hh, e), Te(Zv, e);
    }
    function We() {
      var e = G;
      mu === null ? mu = [e] : mu.push(e);
    }
    function ee() {
      var e = G;
      if (mu !== null && (Qc++, mu[Qc] !== e)) {
        var t = de(Be);
        if (!g1.has(t) && (g1.add(t), mu !== null)) {
          for (var a = "", i = 0; i <= Qc; i++) {
            var o = mu[i], f = i === Qc ? e : o;
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
      e == null || qe(e) || console.error(
        "%s received a final argument that is not an array (instead, received `%s`). When specified, the final argument must be an array.",
        G,
        typeof e
      );
    }
    function po() {
      var e = de(Be);
      S1.has(e) || (S1.add(e), console.error(
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
        if (!Ha(e[a], t[a])) return !1;
      return !0;
    }
    function yi(e, t, a, i, o, f) {
      cf = f, Be = t, mu = e !== null ? e._debugHookTypes : null, Qc = -1, lp = e !== null && e.type !== t.type, (Object.prototype.toString.call(a) === "[object AsyncFunction]" || Object.prototype.toString.call(a) === "[object AsyncGeneratorFunction]") && (f = de(Be), n0.has(f) || (n0.add(f), console.error(
        "%s is an async Client Component. Only Server Components can be async at the moment. This error is often caused by accidentally adding `'use client'` to a module that was originally written for the server.",
        f === null ? "An unknown Component" : "<" + f + ">"
      ))), t.memoizedState = null, t.updateQueue = null, t.lanes = 0, j.H = e !== null && e.memoizedState !== null ? i0 : mu !== null ? T1 : u0, kr = f = (t.mode & Sa) !== Yt;
      var d = c0(a, i, o);
      if (kr = !1, mh && (d = vo(
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
      return Vf(e, t), d;
    }
    function Vf(e, t) {
      t._debugHookTypes = mu, t.dependencies === null ? Xc !== null && (t.dependencies = {
        lanes: 0,
        firstContext: null,
        _debugThenableState: Xc
      }) : t.dependencies._debugThenableState = Xc, j.H = kv;
      var a = Ct !== null && Ct.next !== null;
      if (cf = 0, mu = G = Al = Ct = Be = null, Qc = -1, e !== null && (e.flags & 65011712) !== (t.flags & 65011712) && console.error(
        "Internal React error: Expected static flag was missing. Please notify the React team."
      ), Kv = !1, tp = 0, Xc = null, a)
        throw Error(
          "Rendered fewer hooks than expected. This may be caused by an accidental early return statement."
        );
      e === null || Kl || (e = e.dependencies, e !== null && oi(e) && (Kl = !0)), Qv ? (Qv = !1, e = !0) : e = !1, e && (t = de(t) || "Unknown", b1.has(t) || n0.has(t) || (b1.add(t), console.error(
        "`use` was called from inside a try/catch block. This is not allowed and can lead to unexpected behavior. To handle errors triggered by `use`, wrap your component in a error boundary."
      )));
    }
    function vo(e, t, a, i) {
      Be = e;
      var o = 0;
      do {
        if (mh && (Xc = null), tp = 0, mh = !1, o >= VS)
          throw Error(
            "Too many re-renders. React limits the number of renders to prevent an infinite loop."
          );
        if (o += 1, lp = !1, Al = Ct = null, e.updateQueue != null) {
          var f = e.updateQueue;
          f.lastEffect = null, f.events = null, f.stores = null, f.memoCache != null && (f.memoCache.index = 0);
        }
        Qc = -1, j.H = E1, f = c0(t, a, i);
      } while (mh);
      return f;
    }
    function Xa() {
      var e = j.H, t = e.useState()[0];
      return t = typeof t.then == "function" ? rc(t) : t, e = e.useState()[0], (Ct !== null ? Ct.memoizedState : null) !== e && (Be.flags |= 1024), t;
    }
    function fa() {
      var e = Jv !== 0;
      return Jv = 0, e;
    }
    function zu(e, t, a) {
      t.updateQueue = e.updateQueue, t.flags = (t.mode & Ju) !== Yt ? t.flags & -402655237 : t.flags & -2053, e.lanes &= ~a;
    }
    function mn(e) {
      if (Kv) {
        for (e = e.memoizedState; e !== null; ) {
          var t = e.queue;
          t !== null && (t.pending = null), e = e.next;
        }
        Kv = !1;
      }
      cf = 0, mu = Al = Ct = Be = null, Qc = -1, G = null, mh = !1, tp = Jv = 0, Xc = null;
    }
    function Qt() {
      var e = {
        memoizedState: null,
        baseState: null,
        baseQueue: null,
        queue: null,
        next: null
      };
      return Al === null ? Be.memoizedState = Al = e : Al = Al.next = e, Al;
    }
    function ot() {
      if (Ct === null) {
        var e = Be.alternate;
        e = e !== null ? e.memoizedState : null;
      } else e = Ct.next;
      var t = Al === null ? Be.memoizedState : Al.next;
      if (t !== null)
        Al = t, Ct = e;
      else {
        if (e === null)
          throw Be.alternate === null ? Error(
            "Update hook called on initial render. This is likely a bug in React. Please file an issue."
          ) : Error("Rendered more hooks than during the previous render.");
        Ct = e, e = {
          memoizedState: Ct.memoizedState,
          baseState: Ct.baseState,
          baseQueue: Ct.baseQueue,
          queue: Ct.queue,
          next: null
        }, Al === null ? Be.memoizedState = Al = e : Al = Al.next = e;
      }
      return Al;
    }
    function Cs() {
      return { lastEffect: null, events: null, stores: null, memoCache: null };
    }
    function rc(e) {
      var t = tp;
      return tp += 1, Xc === null && (Xc = oy()), e = La(Xc, e, t), t = Be, (Al === null ? t.memoizedState : Al.next) === null && (t = t.alternate, j.H = t !== null && t.memoizedState !== null ? i0 : u0), e;
    }
    function Yn(e) {
      if (e !== null && typeof e == "object") {
        if (typeof e.then == "function") return rc(e);
        if (e.$$typeof === Ia) return xt(e);
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
          a[i] = Av;
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
    function st(e, t, a) {
      var i = Qt();
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
    function Qa(e) {
      var t = ot();
      return Za(t, Ct, e);
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
          var L = b.lane & -536870913;
          if (L !== b.lane ? (it & L) === L : (cf & L) === L) {
            var H = b.revertLane;
            if (H === 0)
              v !== null && (v = v.next = {
                lane: 0,
                revertLane: 0,
                action: b.action,
                hasEagerState: b.hasEagerState,
                eagerState: b.eagerState,
                next: null
              }), L === Zr && (B = !0);
            else if ((cf & H) === H) {
              b = b.next, H === Zr && (B = !0);
              continue;
            } else
              L = {
                lane: 0,
                revertLane: b.revertLane,
                action: b.action,
                hasEagerState: b.hasEagerState,
                eagerState: b.eagerState,
                next: null
              }, v === null ? (h = v = L, d = f) : v = v.next = L, Be.lanes |= H, rf |= H;
            L = b.action, kr && a(f, L), f = b.hasEagerState ? b.eagerState : a(f, L);
          } else
            H = {
              lane: L,
              revertLane: b.revertLane,
              action: b.action,
              hasEagerState: b.hasEagerState,
              eagerState: b.eagerState,
              next: null
            }, v === null ? (h = v = H, d = f) : v = v.next = H, Be.lanes |= L, rf |= L;
          b = b.next;
        } while (b !== null && b !== t);
        if (v === null ? d = f : v.next = h, !Ha(f, e.memoizedState) && (Kl = !0, B && (a = dh, a !== null)))
          throw a;
        e.memoizedState = f, e.baseState = d, e.baseQueue = v, i.lastRenderedState = f;
      }
      return o === null && (i.lanes = 0), [e.memoizedState, i.dispatch];
    }
    function sc(e) {
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
    function Mu(e, t, a) {
      var i = Be, o = Qt();
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
        if (f = t(), yh || (a = t(), Ha(f, a) || (console.error(
          "The result of getSnapshot should be cached to avoid an infinite loop"
        ), yh = !0)), Ht === null)
          throw Error(
            "Expected a work-in-progress root. This is a bug in React. Please file an issue."
          );
        (it & 124) !== 0 || sy(i, t, f);
      }
      return o.memoizedState = f, a = { value: f, getSnapshot: t }, o.queue = a, Ns(
        bo.bind(null, i, a, e),
        [e]
      ), i.flags |= 2048, Gn(
        yu | Yl,
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
    function Xf(e, t, a) {
      var i = Be, o = ot(), f = mt;
      if (f) {
        if (a === void 0)
          throw Error(
            "Missing getServerSnapshot, which is required for server-rendered content. Will revert to client rendering."
          );
        a = a();
      } else if (a = t(), !yh) {
        var d = t();
        Ha(a, d) || (console.error(
          "The result of getSnapshot should be cached to avoid an infinite loop"
        ), yh = !0);
      }
      (d = !Ha(
        (Ct || o).memoizedState,
        a
      )) && (o.memoizedState = a, Kl = !0), o = o.queue;
      var h = bo.bind(null, i, o, e);
      if (rl(2048, Yl, h, [e]), o.getSnapshot !== t || d || Al !== null && Al.memoizedState.tag & yu) {
        if (i.flags |= 2048, Gn(
          yu | Yl,
          pi(),
          go.bind(
            null,
            i,
            o,
            a,
            t
          ),
          null
        ), Ht === null)
          throw Error(
            "Expected a work-in-progress root. This is a bug in React. Please file an issue."
          );
        f || (cf & 124) !== 0 || sy(i, t, a);
      }
      return a;
    }
    function sy(e, t, a) {
      e.flags |= 16384, e = { getSnapshot: t, value: a }, t = Be.updateQueue, t === null ? (t = Cs(), Be.updateQueue = t, t.stores = [e]) : (a = t.stores, a === null ? t.stores = [e] : a.push(e));
    }
    function go(e, t, a, i) {
      t.value = a, t.getSnapshot = i, dy(t) && So(e);
    }
    function bo(e, t, a) {
      return a(function() {
        dy(t) && So(e);
      });
    }
    function dy(e) {
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
    function Qf(e) {
      var t = Qt();
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
      var t = e.queue, a = Ro.bind(null, Be, t);
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
      return t.queue = a, t = Xs.bind(
        null,
        Be,
        !0,
        a
      ), a.dispatch = t, [e, t];
    }
    function _u(e, t) {
      var a = ot();
      return jn(a, Ct, e, t);
    }
    function jn(e, t, a, i) {
      return e.baseState = a, Za(
        e,
        Ct,
        typeof i == "function" ? i : dt
      );
    }
    function xs(e, t) {
      var a = ot();
      return Ct !== null ? jn(a, Ct, e, t) : (a.baseState = e, [e, a.queue.dispatch]);
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
        j.T !== null ? a(!0) : f.isTransition = !1, i(f), a = t.pending, a === null ? (f.next = t.pending = f, To(t, f)) : (f.next = a.next, t.pending = a.next = f);
      }
    }
    function To(e, t) {
      var a = t.action, i = t.payload, o = e.state;
      if (t.isTransition) {
        var f = j.T, d = {};
        j.T = d, j.T._updatedFibers = /* @__PURE__ */ new Set();
        try {
          var h = a(o, i), v = j.S;
          v !== null && v(d, h), Zf(e, t, h);
        } catch (b) {
          gl(e, t, b);
        } finally {
          j.T = f, f === null && d._updatedFibers && (e = d._updatedFibers.size, d._updatedFibers.clear(), 10 < e && console.warn(
            "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
          ));
        }
      } else
        try {
          d = a(o, i), Zf(e, t, d);
        } catch (b) {
          gl(e, t, b);
        }
    }
    function Zf(e, t, a) {
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
      t.status = "fulfilled", t.value = a, Kf(t), e.state = a, t = e.pending, t !== null && (a = t.next, a === t ? e.pending = null : (a = a.next, t.next = a, To(e, a)));
    }
    function gl(e, t, a) {
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
    function Eo(e, t) {
      if (mt) {
        var a = Ht.formState;
        if (a !== null) {
          e: {
            var i = Be;
            if (mt) {
              if (nl) {
                t: {
                  for (var o = nl, f = ji; o.nodeType !== 8; ) {
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
                  f = o.data, o = f === H0 || f === Sb ? o : null;
                }
                if (o) {
                  nl = Nl(
                    o.nextSibling
                  ), i = o.data === H0;
                  break e;
                }
              }
              Hn(i);
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
        lastRenderedReducer: yy,
        lastRenderedState: t
      }, a.queue = i, a = Ro.bind(
        null,
        Be,
        i
      ), i.dispatch = a, i = Qf(!1), f = Xs.bind(
        null,
        Be,
        !1,
        i.queue
      ), i = Qt(), o = {
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
      var t = ot();
      return $p(t, Ct, e);
    }
    function $p(e, t, a) {
      if (t = Za(
        e,
        t,
        yy
      )[0], e = Qa(dt)[0], typeof t == "object" && t !== null && typeof t.then == "function")
        try {
          var i = rc(t);
        } catch (d) {
          throw d === Pm ? Xv : d;
        }
      else i = t;
      t = ot();
      var o = t.queue, f = o.dispatch;
      return a !== t.memoizedState && (Be.flags |= 2048, Gn(
        yu | Yl,
        pi(),
        fl.bind(null, o, a),
        null
      )), [i, f, e];
    }
    function fl(e, t) {
      e.action = t;
    }
    function Ao(e) {
      var t = ot(), a = Ct;
      if (a !== null)
        return $p(t, a, e);
      ot(), t = t.memoizedState, a = ot();
      var i = a.queue.dispatch;
      return a.memoizedState = e, [t, i, !1];
    }
    function Gn(e, t, a, i) {
      return e = {
        tag: e,
        create: a,
        deps: i,
        inst: t,
        next: null
      }, t = Be.updateQueue, t === null && (t = Cs(), Be.updateQueue = t), a = t.lastEffect, a === null ? t.lastEffect = e.next = e : (i = a.next, a.next = e, e.next = i, t.lastEffect = e), e;
    }
    function pi() {
      return { destroy: void 0, resource: void 0 };
    }
    function Jf(e) {
      var t = Qt();
      return e = { current: e }, t.memoizedState = e;
    }
    function Ka(e, t, a, i) {
      var o = Qt();
      i = i === void 0 ? null : i, Be.flags |= e, o.memoizedState = Gn(
        yu | t,
        pi(),
        a,
        i
      );
    }
    function rl(e, t, a, i) {
      var o = ot();
      i = i === void 0 ? null : i;
      var f = o.memoizedState.inst;
      Ct !== null && i !== null && hi(i, Ct.memoizedState.deps) ? o.memoizedState = Gn(t, f, a, i) : (Be.flags |= e, o.memoizedState = Gn(
        yu | t,
        f,
        a,
        i
      ));
    }
    function Ns(e, t) {
      (Be.mode & Ju) !== Yt && (Be.mode & n1) === Yt ? Ka(276826112, Yl, e, t) : Ka(8390656, Yl, e, t);
    }
    function ws(e, t) {
      var a = 4194308;
      return (Be.mode & Ju) !== Yt && (a |= 134217728), Ka(a, la, e, t);
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
      (Be.mode & Ju) !== Yt && (i |= 134217728), Ka(
        i,
        la,
        Wp.bind(null, t, e),
        a
      );
    }
    function Ln(e, t, a) {
      typeof t != "function" && console.error(
        "Expected useImperativeHandle() second argument to be a function that creates a handle. Instead received: %s.",
        t !== null ? typeof t : "null"
      ), a = a != null ? a.concat([e]) : null, rl(
        4,
        la,
        Wp.bind(null, t, e),
        a
      );
    }
    function kf(e, t) {
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
    function Bs(e, t) {
      var a = Qt();
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
    function vi(e, t) {
      var a = ot();
      t = t === void 0 ? null : t;
      var i = a.memoizedState;
      if (t !== null && hi(t, i[1]))
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
      var a = Qt();
      return Gs(a, e, t);
    }
    function $f(e, t) {
      var a = ot();
      return Wf(
        a,
        Ct.memoizedState,
        e,
        t
      );
    }
    function js(e, t) {
      var a = ot();
      return Ct === null ? Gs(a, e, t) : Wf(
        a,
        Ct.memoizedState,
        e,
        t
      );
    }
    function Gs(e, t, a) {
      return a === void 0 || (cf & 1073741824) !== 0 ? e.memoizedState = t : (e.memoizedState = a, e = iv(), Be.lanes |= e, rf |= e, a);
    }
    function Wf(e, t, a, i) {
      return Ha(a, t) ? a : hh.current !== null ? (e = Gs(e, a, i), Ha(e, t) || (Kl = !0), e) : (cf & 42) === 0 ? (Kl = !0, e.memoizedState = a) : (e = iv(), Be.lanes |= e, rf |= e, t);
    }
    function my(e, t, a, i, o) {
      var f = xe.p;
      xe.p = f !== 0 && f < En ? f : En;
      var d = j.T, h = {};
      j.T = h, Xs(e, !1, t, a), h._updatedFibers = /* @__PURE__ */ new Set();
      try {
        var v = o(), b = j.S;
        if (b !== null && b(h, v), v !== null && typeof v == "object" && typeof v.then == "function") {
          var B = Kp(
            v,
            i
          );
          Cu(
            e,
            t,
            B,
            ha(e)
          );
        } else
          Cu(
            e,
            t,
            i,
            ha(e)
          );
      } catch (L) {
        Cu(
          e,
          t,
          { then: function() {
          }, status: "rejected", reason: L },
          ha(e)
        );
      } finally {
        xe.p = f, j.T = d, d === null && h._updatedFibers && (e = h._updatedFibers.size, h._updatedFibers.clear(), 10 < e && console.warn(
          "Detected a large number of updates inside startTransition. If this is due to a subscription please re-write it to use React provided hooks. Otherwise concurrent mode guarantees are off the table."
        ));
      }
    }
    function hc(e, t, a, i) {
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
        a === null ? le : function() {
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
      j.T === null && console.error(
        "requestFormReset was called outside a transition or action. To fix, move to an action, or wrap with startTransition."
      );
      var t = py(e).next.queue;
      Cu(
        e,
        t,
        {},
        ha(e)
      );
    }
    function Vn() {
      var e = Qf(!1);
      return e = my.bind(
        null,
        Be,
        e.queue,
        !0,
        !1
      ), Qt().memoizedState = e, [!1, e];
    }
    function Ls() {
      var e = Qa(dt)[0], t = ot().memoizedState;
      return [
        typeof e == "boolean" ? e : rc(e),
        t
      ];
    }
    function Vs() {
      var e = sc(dt)[0], t = ot().memoizedState;
      return [
        typeof e == "boolean" ? e : rc(e),
        t
      ];
    }
    function ra() {
      return xt(bp);
    }
    function Xn() {
      var e = Qt(), t = Ht.identifierPrefix;
      if (mt) {
        var a = Lc, i = Gc;
        a = (i & ~(1 << 32 - Ql(i) - 1)).toString(32) + a, t = "«" + t + "R" + a, a = Jv++, 0 < a && (t += "H" + a.toString(32)), t += "»";
      } else
        a = LS++, t = "«" + t + "r" + a.toString(32) + "»";
      return e.memoizedState = t;
    }
    function yc() {
      return Qt().memoizedState = gy.bind(
        null,
        Be
      );
    }
    function gy(e, t) {
      for (var a = e.return; a !== null; ) {
        switch (a.tag) {
          case 24:
          case 3:
            var i = ha(a);
            e = qn(i);
            var o = hn(a, e, i);
            o !== null && (Kt(o, a, i), di(o, a, i)), a = jf(), t != null && o !== null && console.error(
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
      ), i = ha(e);
      var o = {
        lane: i,
        revertLane: 0,
        action: a,
        hasEagerState: !1,
        eagerState: null,
        next: null
      };
      Ff(e) ? mc(t, o) : (o = Jh(e, t, o, i), o !== null && (Kt(o, e, i), If(o, t, i))), Mn(e, i);
    }
    function Ro(e, t, a) {
      var i = arguments;
      typeof i[3] == "function" && console.error(
        "State updates from the useState() and useReducer() Hooks don't support the second callback argument. To execute a side effect after rendering, declare it in the component body with useEffect()."
      ), i = ha(e), Cu(e, t, a, i), Mn(e, i);
    }
    function Cu(e, t, a, i) {
      var o = {
        lane: i,
        revertLane: 0,
        action: a,
        hasEagerState: !1,
        eagerState: null,
        next: null
      };
      if (Ff(e)) mc(t, o);
      else {
        var f = e.alternate;
        if (e.lanes === 0 && (f === null || f.lanes === 0) && (f = t.lastRenderedReducer, f !== null)) {
          var d = j.H;
          j.H = $u;
          try {
            var h = t.lastRenderedState, v = f(h, a);
            if (o.hasEagerState = !0, o.eagerState = v, Ha(v, h))
              return As(e, t, o, 0), Ht === null && Nf(), !1;
          } catch {
          } finally {
            j.H = d;
          }
        }
        if (a = Jh(e, t, o, i), a !== null)
          return Kt(a, e, i), If(a, t, i), !0;
      }
      return !1;
    }
    function Xs(e, t, a, i) {
      if (j.T === null && Zr === 0 && console.error(
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
        ), t !== null && Kt(t, e, 2);
      Mn(e, 2);
    }
    function Ff(e) {
      var t = e.alternate;
      return e === Be || t !== null && t === Be;
    }
    function mc(e, t) {
      mh = Kv = !0;
      var a = e.pending;
      a === null ? t.next = t : (t.next = a.next, a.next = t), e.pending = t;
    }
    function If(e, t, a) {
      if ((a & 4194048) !== 0) {
        var i = t.lanes;
        i &= e.pendingLanes, a |= i, t.lanes = a, Pu(e, a);
      }
    }
    function bl(e) {
      var t = Ie;
      return e != null && (Ie = t === null ? e : t.concat(e)), t;
    }
    function Oo(e, t, a) {
      for (var i = Object.keys(e.props), o = 0; o < i.length; o++) {
        var f = i[o];
        if (f !== "children" && f !== "key") {
          t === null && (t = Bf(e, a.mode, 0), t._debugInfo = Ie, t.return = a), ye(
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
      var t = ap;
      return ap += 1, ph === null && (ph = oy()), La(ph, e, t);
    }
    function Ja(e, t) {
      t = t.props.ref, e.ref = t !== void 0 ? t : null;
    }
    function Ve(e, t) {
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
    function Vt(e, t) {
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
        return T = xn(T, E), T.index = 0, T.sibling = null, T;
      }
      function f(T, E, A) {
        return T.index = A, e ? (A = T.alternate, A !== null ? (A = A.index, A < E ? (T.flags |= 67108866, E) : A) : (T.flags |= 67108866, E)) : (T.flags |= 1048576, E);
      }
      function d(T) {
        return e && T.alternate === null && (T.flags |= 67108866), T;
      }
      function h(T, E, A, Q) {
        return E === null || E.tag !== 6 ? (E = ii(
          A,
          T.mode,
          Q
        ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Ie, E) : (E = o(E, A), E.return = T, E._debugInfo = Ie, E);
      }
      function v(T, E, A, Q) {
        var ne = A.type;
        return ne === Xe ? (E = B(
          T,
          E,
          A.props.children,
          Q,
          A.key
        ), Oo(A, E, T), E) : E !== null && (E.elementType === ne || Vp(E, A) || typeof ne == "object" && ne !== null && ne.$$typeof === Ca && of(ne) === E.type) ? (E = o(E, A.props), Ja(E, A), E.return = T, E._debugOwner = A._owner, E._debugInfo = Ie, E) : (E = Bf(A, T.mode, Q), Ja(E, A), E.return = T, E._debugInfo = Ie, E);
      }
      function b(T, E, A, Q) {
        return E === null || E.tag !== 4 || E.stateNode.containerInfo !== A.containerInfo || E.stateNode.implementation !== A.implementation ? (E = Fh(A, T.mode, Q), E.return = T, E._debugInfo = Ie, E) : (E = o(E, A.children || []), E.return = T, E._debugInfo = Ie, E);
      }
      function B(T, E, A, Q, ne) {
        return E === null || E.tag !== 7 ? (E = ui(
          A,
          T.mode,
          Q,
          ne
        ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Ie, E) : (E = o(E, A), E.return = T, E._debugInfo = Ie, E);
      }
      function L(T, E, A) {
        if (typeof E == "string" && E !== "" || typeof E == "number" || typeof E == "bigint")
          return E = ii(
            "" + E,
            T.mode,
            A
          ), E.return = T, E._debugOwner = T, E._debugTask = T._debugTask, E._debugInfo = Ie, E;
        if (typeof E == "object" && E !== null) {
          switch (E.$$typeof) {
            case _i:
              return A = Bf(
                E,
                T.mode,
                A
              ), Ja(A, E), A.return = T, T = bl(E._debugInfo), A._debugInfo = Ie, Ie = T, A;
            case Nc:
              return E = Fh(
                E,
                T.mode,
                A
              ), E.return = T, E._debugInfo = Ie, E;
            case Ca:
              var Q = bl(E._debugInfo);
              return E = of(E), T = L(T, E, A), Ie = Q, T;
          }
          if (qe(E) || bt(E))
            return A = ui(
              E,
              T.mode,
              A,
              null
            ), A.return = T, A._debugOwner = T, A._debugTask = T._debugTask, T = bl(E._debugInfo), A._debugInfo = Ie, Ie = T, A;
          if (typeof E.then == "function")
            return Q = bl(E._debugInfo), T = L(
              T,
              Do(E),
              A
            ), Ie = Q, T;
          if (E.$$typeof === Ia)
            return L(
              T,
              Yf(T, E),
              A
            );
          Ve(T, E);
        }
        return typeof E == "function" && vt(T, E), typeof E == "symbol" && Vt(T, E), null;
      }
      function H(T, E, A, Q) {
        var ne = E !== null ? E.key : null;
        if (typeof A == "string" && A !== "" || typeof A == "number" || typeof A == "bigint")
          return ne !== null ? null : h(T, E, "" + A, Q);
        if (typeof A == "object" && A !== null) {
          switch (A.$$typeof) {
            case _i:
              return A.key === ne ? (ne = bl(A._debugInfo), T = v(
                T,
                E,
                A,
                Q
              ), Ie = ne, T) : null;
            case Nc:
              return A.key === ne ? b(T, E, A, Q) : null;
            case Ca:
              return ne = bl(A._debugInfo), A = of(A), T = H(
                T,
                E,
                A,
                Q
              ), Ie = ne, T;
          }
          if (qe(A) || bt(A))
            return ne !== null ? null : (ne = bl(A._debugInfo), T = B(
              T,
              E,
              A,
              Q,
              null
            ), Ie = ne, T);
          if (typeof A.then == "function")
            return ne = bl(A._debugInfo), T = H(
              T,
              E,
              Do(A),
              Q
            ), Ie = ne, T;
          if (A.$$typeof === Ia)
            return H(
              T,
              E,
              Yf(T, A),
              Q
            );
          Ve(T, A);
        }
        return typeof A == "function" && vt(T, A), typeof A == "symbol" && Vt(T, A), null;
      }
      function X(T, E, A, Q, ne) {
        if (typeof Q == "string" && Q !== "" || typeof Q == "number" || typeof Q == "bigint")
          return T = T.get(A) || null, h(E, T, "" + Q, ne);
        if (typeof Q == "object" && Q !== null) {
          switch (Q.$$typeof) {
            case _i:
              return A = T.get(
                Q.key === null ? A : Q.key
              ) || null, T = bl(Q._debugInfo), E = v(
                E,
                A,
                Q,
                ne
              ), Ie = T, E;
            case Nc:
              return T = T.get(
                Q.key === null ? A : Q.key
              ) || null, b(E, T, Q, ne);
            case Ca:
              var Qe = bl(Q._debugInfo);
              return Q = of(Q), E = X(
                T,
                E,
                A,
                Q,
                ne
              ), Ie = Qe, E;
          }
          if (qe(Q) || bt(Q))
            return A = T.get(A) || null, T = bl(Q._debugInfo), E = B(
              E,
              A,
              Q,
              ne,
              null
            ), Ie = T, E;
          if (typeof Q.then == "function")
            return Qe = bl(Q._debugInfo), E = X(
              T,
              E,
              A,
              Do(Q),
              ne
            ), Ie = Qe, E;
          if (Q.$$typeof === Ia)
            return X(
              T,
              E,
              A,
              Yf(E, Q),
              ne
            );
          Ve(E, Q);
        }
        return typeof Q == "function" && vt(E, Q), typeof Q == "symbol" && Vt(E, Q), null;
      }
      function me(T, E, A, Q) {
        if (typeof A != "object" || A === null) return Q;
        switch (A.$$typeof) {
          case _i:
          case Nc:
            Se(T, E, A);
            var ne = A.key;
            if (typeof ne != "string") break;
            if (Q === null) {
              Q = /* @__PURE__ */ new Set(), Q.add(ne);
              break;
            }
            if (!Q.has(ne)) {
              Q.add(ne);
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
            A = of(A), me(T, E, A, Q);
        }
        return Q;
      }
      function He(T, E, A, Q) {
        for (var ne = null, Qe = null, pe = null, Ze = E, Ke = E = 0, jt = null; Ze !== null && Ke < A.length; Ke++) {
          Ze.index > Ke ? (jt = Ze, Ze = null) : jt = Ze.sibling;
          var yl = H(
            T,
            Ze,
            A[Ke],
            Q
          );
          if (yl === null) {
            Ze === null && (Ze = jt);
            break;
          }
          ne = me(
            T,
            yl,
            A[Ke],
            ne
          ), e && Ze && yl.alternate === null && t(T, Ze), E = f(yl, E, Ke), pe === null ? Qe = yl : pe.sibling = yl, pe = yl, Ze = jt;
        }
        if (Ke === A.length)
          return a(T, Ze), mt && nc(T, Ke), Qe;
        if (Ze === null) {
          for (; Ke < A.length; Ke++)
            Ze = L(T, A[Ke], Q), Ze !== null && (ne = me(
              T,
              Ze,
              A[Ke],
              ne
            ), E = f(
              Ze,
              E,
              Ke
            ), pe === null ? Qe = Ze : pe.sibling = Ze, pe = Ze);
          return mt && nc(T, Ke), Qe;
        }
        for (Ze = i(Ze); Ke < A.length; Ke++)
          jt = X(
            Ze,
            T,
            Ke,
            A[Ke],
            Q
          ), jt !== null && (ne = me(
            T,
            jt,
            A[Ke],
            ne
          ), e && jt.alternate !== null && Ze.delete(
            jt.key === null ? Ke : jt.key
          ), E = f(
            jt,
            E,
            Ke
          ), pe === null ? Qe = jt : pe.sibling = jt, pe = jt);
        return e && Ze.forEach(function(Wc) {
          return t(T, Wc);
        }), mt && nc(T, Ke), Qe;
      }
      function Nt(T, E, A, Q) {
        if (A == null)
          throw Error("An iterable object provided no iterator.");
        for (var ne = null, Qe = null, pe = E, Ze = E = 0, Ke = null, jt = null, yl = A.next(); pe !== null && !yl.done; Ze++, yl = A.next()) {
          pe.index > Ze ? (Ke = pe, pe = null) : Ke = pe.sibling;
          var Wc = H(T, pe, yl.value, Q);
          if (Wc === null) {
            pe === null && (pe = Ke);
            break;
          }
          jt = me(
            T,
            Wc,
            yl.value,
            jt
          ), e && pe && Wc.alternate === null && t(T, pe), E = f(Wc, E, Ze), Qe === null ? ne = Wc : Qe.sibling = Wc, Qe = Wc, pe = Ke;
        }
        if (yl.done)
          return a(T, pe), mt && nc(T, Ze), ne;
        if (pe === null) {
          for (; !yl.done; Ze++, yl = A.next())
            pe = L(T, yl.value, Q), pe !== null && (jt = me(
              T,
              pe,
              yl.value,
              jt
            ), E = f(
              pe,
              E,
              Ze
            ), Qe === null ? ne = pe : Qe.sibling = pe, Qe = pe);
          return mt && nc(T, Ze), ne;
        }
        for (pe = i(pe); !yl.done; Ze++, yl = A.next())
          Ke = X(
            pe,
            T,
            Ze,
            yl.value,
            Q
          ), Ke !== null && (jt = me(
            T,
            Ke,
            yl.value,
            jt
          ), e && Ke.alternate !== null && pe.delete(
            Ke.key === null ? Ze : Ke.key
          ), E = f(
            Ke,
            E,
            Ze
          ), Qe === null ? ne = Ke : Qe.sibling = Ke, Qe = Ke);
        return e && pe.forEach(function(yT) {
          return t(T, yT);
        }), mt && nc(T, Ze), ne;
      }
      function ft(T, E, A, Q) {
        if (typeof A == "object" && A !== null && A.type === Xe && A.key === null && (Oo(A, null, T), A = A.props.children), typeof A == "object" && A !== null) {
          switch (A.$$typeof) {
            case _i:
              var ne = bl(A._debugInfo);
              e: {
                for (var Qe = A.key; E !== null; ) {
                  if (E.key === Qe) {
                    if (Qe = A.type, Qe === Xe) {
                      if (E.tag === 7) {
                        a(
                          T,
                          E.sibling
                        ), Q = o(
                          E,
                          A.props.children
                        ), Q.return = T, Q._debugOwner = A._owner, Q._debugInfo = Ie, Oo(A, Q, T), T = Q;
                        break e;
                      }
                    } else if (E.elementType === Qe || Vp(
                      E,
                      A
                    ) || typeof Qe == "object" && Qe !== null && Qe.$$typeof === Ca && of(Qe) === E.type) {
                      a(
                        T,
                        E.sibling
                      ), Q = o(E, A.props), Ja(Q, A), Q.return = T, Q._debugOwner = A._owner, Q._debugInfo = Ie, T = Q;
                      break e;
                    }
                    a(T, E);
                    break;
                  } else t(T, E);
                  E = E.sibling;
                }
                A.type === Xe ? (Q = ui(
                  A.props.children,
                  T.mode,
                  Q,
                  A.key
                ), Q.return = T, Q._debugOwner = T, Q._debugTask = T._debugTask, Q._debugInfo = Ie, Oo(A, Q, T), T = Q) : (Q = Bf(
                  A,
                  T.mode,
                  Q
                ), Ja(Q, A), Q.return = T, Q._debugInfo = Ie, T = Q);
              }
              return T = d(T), Ie = ne, T;
            case Nc:
              e: {
                for (ne = A, A = ne.key; E !== null; ) {
                  if (E.key === A)
                    if (E.tag === 4 && E.stateNode.containerInfo === ne.containerInfo && E.stateNode.implementation === ne.implementation) {
                      a(
                        T,
                        E.sibling
                      ), Q = o(
                        E,
                        ne.children || []
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
                  ne,
                  T.mode,
                  Q
                ), Q.return = T, T = Q;
              }
              return d(T);
            case Ca:
              return ne = bl(A._debugInfo), A = of(A), T = ft(
                T,
                E,
                A,
                Q
              ), Ie = ne, T;
          }
          if (qe(A))
            return ne = bl(A._debugInfo), T = He(
              T,
              E,
              A,
              Q
            ), Ie = ne, T;
          if (bt(A)) {
            if (ne = bl(A._debugInfo), Qe = bt(A), typeof Qe != "function")
              throw Error(
                "An object is not an iterable. This error is likely caused by a bug in React. Please file an issue."
              );
            var pe = Qe.call(A);
            return pe === A ? (T.tag !== 0 || Object.prototype.toString.call(T.type) !== "[object GeneratorFunction]" || Object.prototype.toString.call(pe) !== "[object Generator]") && (w1 || console.error(
              "Using Iterators as children is unsupported and will likely yield unexpected results because enumerating a generator mutates it. You may convert it to an array with `Array.from()` or the `[...spread]` operator before rendering. You can also use an Iterable that can iterate multiple times over the same items."
            ), w1 = !0) : A.entries !== Qe || f0 || (console.error(
              "Using Maps as children is not supported. Use an array of keyed ReactElements instead."
            ), f0 = !0), T = Nt(
              T,
              E,
              pe,
              Q
            ), Ie = ne, T;
          }
          if (typeof A.then == "function")
            return ne = bl(A._debugInfo), T = ft(
              T,
              E,
              Do(A),
              Q
            ), Ie = ne, T;
          if (A.$$typeof === Ia)
            return ft(
              T,
              E,
              Yf(T, A),
              Q
            );
          Ve(T, A);
        }
        return typeof A == "string" && A !== "" || typeof A == "number" || typeof A == "bigint" ? (ne = "" + A, E !== null && E.tag === 6 ? (a(
          T,
          E.sibling
        ), Q = o(E, ne), Q.return = T, T = Q) : (a(T, E), Q = ii(
          ne,
          T.mode,
          Q
        ), Q.return = T, Q._debugOwner = T, Q._debugTask = T._debugTask, Q._debugInfo = Ie, T = Q), d(T)) : (typeof A == "function" && vt(T, A), typeof A == "symbol" && Vt(T, A), a(T, E));
      }
      return function(T, E, A, Q) {
        var ne = Ie;
        Ie = null;
        try {
          ap = 0;
          var Qe = ft(
            T,
            E,
            A,
            Q
          );
          return ph = null, Qe;
        } catch (jt) {
          if (jt === Pm || jt === Xv) throw jt;
          var pe = U(29, jt, null, T.mode);
          pe.lanes = Q, pe.return = T;
          var Ze = pe._debugInfo = Ie;
          if (pe._debugOwner = T._debugOwner, pe._debugTask = T._debugTask, Ze != null) {
            for (var Ke = Ze.length - 1; 0 <= Ke; Ke--)
              if (typeof Ze[Ke].stack == "string") {
                pe._debugOwner = Ze[Ke], pe._debugTask = Ze[Ke].debugTask;
                break;
              }
          }
          return pe;
        } finally {
          Ie = ne;
        }
      };
    }
    function Da(e) {
      var t = e.alternate;
      Ce(
        jl,
        jl.current & gh,
        e
      ), Ce(pu, e, e), Li === null && (t === null || hh.current !== null || t.memoizedState !== null) && (Li = e);
    }
    function gi(e) {
      if (e.tag === 22) {
        if (Ce(jl, jl.current, e), Ce(pu, e, e), Li === null) {
          var t = e.alternate;
          t !== null && t.memoizedState !== null && (Li = e);
        }
      } else vn(e);
    }
    function vn(e) {
      Ce(jl, jl.current, e), Ce(
        pu,
        pu.current,
        e
      );
    }
    function za(e) {
      Te(pu, e), Li === e && (Li = null), Te(jl, e);
    }
    function xu(e) {
      for (var t = e; t !== null; ) {
        if (t.tag === 13) {
          var a = t.memoizedState;
          if (a !== null && (a = a.dehydrated, a === null || a.data === Jc || lu(a)))
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
      f === void 0 && (t = Ge(t) || "Component", K1.has(t) || (K1.add(t), console.error(
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
          Ge(t) || "Component"
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
    function bi(e, t) {
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
          Ub,
          hg + i + hg,
          _b
        ) : e.splice(
          0,
          0,
          Mb,
          Ub,
          hg + i + hg,
          _b
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
    function zo(e, t) {
      try {
        bh = t.source ? de(t.source) : null, d0 = null;
        var a = t.value;
        if (j.actQueue !== null)
          j.thrownErrors.push(a);
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
    function Ll(e, t, a) {
      return a = qn(a), a.tag = t0, a.payload = { element: null }, a.callback = function() {
        ye(t.source, zo, e, t);
      }, a;
    }
    function Zt(e) {
      return e = qn(e), e.tag = t0, e;
    }
    function er(e, t, a, i) {
      var o = a.type.getDerivedStateFromError;
      if (typeof o == "function") {
        var f = i.value;
        e.payload = function() {
          return o(f);
        }, e.callback = function() {
          Xp(a), ye(
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
        Xp(a), ye(
          i.source,
          Js,
          t,
          a,
          i
        ), typeof o != "function" && (df === null ? df = /* @__PURE__ */ new Set([this]) : df.add(this)), XS(this, i), typeof o == "function" || (a.lanes & 2) === 0 && console.error(
          "%s: Error boundaries should implement getDerivedStateFromError(). In that method, return a state update to display an error message or fallback UI.",
          de(a) || "Unknown"
        );
      });
    }
    function tr(e, t, a, i, o) {
      if (a.flags |= 32768, Ft && wo(e, o), i !== null && typeof i == "object" && typeof i.then == "function") {
        if (t = a.alternate, t !== null && _l(
          t,
          a,
          o,
          !0
        ), mt && (Vc = !0), a = pu.current, a !== null) {
          switch (a.tag) {
            case 13:
              return Li === null ? hd() : a.alternate === null && ul === Kc && (ul = p0), a.flags &= -257, a.flags |= 65536, a.lanes = o, i === e0 ? a.flags |= 16384 : (t = a.updateQueue, t === null ? a.updateQueue = /* @__PURE__ */ new Set([i]) : t.add(i), Ky(e, i, o)), !1;
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
        return Vc = !0, t = pu.current, t !== null ? ((t.flags & 65536) === 0 && (t.flags |= 256), t.flags |= 65536, t.lanes = o, i !== Wg && ro(
          Ra(
            Error(
              "There was an error while hydrating but React was able to recover by instead client rendering from the nearest Suspense boundary.",
              { cause: i }
            ),
            a
          )
        )) : (i !== Wg && ro(
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
        ), ho(e, o), ul !== $r && (ul = Ah)), !1;
      var f = Ra(
        Error(
          "There was an error during concurrent rendering but React was able to recover by instead synchronously rendering the entire root.",
          { cause: i }
        ),
        a
      );
      if (sp === null ? sp = [f] : sp.push(f), ul !== $r && (ul = Ah), t === null) return !0;
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
              return a.flags |= 65536, o &= -o, a.lanes |= o, o = Zt(o), er(
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
      return fi(t), wt(t), i = yi(
        e,
        t,
        a,
        d,
        f,
        o
      ), h = fa(), na(), e !== null && !Kl ? (zu(e, t, o), Zn(e, t, o)) : (mt && h && Ds(t), t.flags |= 1, al(e, t, i, o), t.child);
    }
    function Qn(e, t, a, i, o) {
      if (e === null) {
        var f = a.type;
        return typeof f == "function" && !$h(f) && f.defaultProps === void 0 && a.compare === null ? (a = ac(f), t.tag = 15, t.type = a, Is(t, f), lr(
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
          return Zn(
            e,
            t,
            o
          );
      }
      return t.flags |= 1, e = xn(f, i), e.ref = t.ref, e.return = t, t.child = e;
    }
    function lr(e, t, a, i, o) {
      if (e !== null) {
        var f = e.memoizedProps;
        if (Hf(f, i) && e.ref === t.ref && t.type === e.type)
          if (Kl = !1, t.pendingProps = i = f, nd(e, o))
            (e.flags & 131072) !== 0 && (Kl = !0);
          else
            return t.lanes = e.lanes, Zn(e, t, o);
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
          t.memoizedState = { baseLanes: 0, cachePool: null }, e !== null && _s(
            t,
            f !== null ? f.cachePool : null
          ), f !== null ? oa(t, f) : Lf(t), gi(t);
        else
          return t.lanes = t.childLanes = 536870912, Ws(
            e,
            t,
            f !== null ? f.baseLanes | a : a,
            a
          );
      } else
        f !== null ? (_s(t, f.cachePool), oa(t, f), vn(t), t.memoizedState = null) : (e !== null && _s(t, null), Lf(t), vn(t));
      return al(e, t, o, a), t.child;
    }
    function Ws(e, t, a, i) {
      var o = cy();
      return o = o === null ? null : {
        parent: Bl._currentValue,
        pool: o
      }, t.memoizedState = {
        baseLanes: a,
        cachePool: o
      }, e !== null && _s(t, null), Lf(t), gi(t), e !== null && _l(e, t, i, !0), null;
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
        var f = Ge(a) || "Unknown";
        I1[f] || (console.error(
          "The <%s /> component appears to have a render method, but doesn't extend React.Component. This is likely to cause errors. Change %s to extend React.Component instead.",
          f,
          f
        ), I1[f] = !0);
      }
      return t.mode & Sa && ku.recordLegacyContextWarning(
        t,
        null
      ), e === null && (Is(t, t.type), a.contextTypes && (f = Ge(a) || "Unknown", eb[f] || (eb[f] = !0, console.error(
        "%s uses the legacy contextTypes API which was removed in React 19. Use React.createContext() with React.useContext() instead. (https://react.dev/link/legacy-context)",
        f
      )))), fi(t), wt(t), a = yi(
        e,
        t,
        a,
        i,
        void 0,
        o
      ), i = fa(), na(), e !== null && !Kl ? (zu(e, t, o), Zn(e, t, o)) : (mt && i && Ds(t), t.flags |= 1, al(e, t, a, o), t.child);
    }
    function Ey(e, t, a, i, o, f) {
      return fi(t), wt(t), Qc = -1, lp = e !== null && e.type !== t.type, t.updateQueue = null, a = vo(
        t,
        i,
        a,
        o
      ), Vf(e, t), i = fa(), na(), e !== null && !Kl ? (zu(e, t, f), Zn(e, t, f)) : (mt && i && Ds(t), t.flags |= 1, al(e, t, a, f), t.child);
    }
    function Ay(e, t, a, i, o) {
      switch (Oe(t)) {
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
          if (t.lanes |= h, d = Ht, d === null)
            throw Error(
              "Expected a work-in-progress root. This is a bug in React. Please file an issue."
            );
          h = Zt(h), er(
            h,
            d,
            t,
            Ra(f, t)
          ), ho(t, h);
      }
      if (fi(t), t.stateNode === null) {
        if (d = nf, f = a.contextType, "contextType" in a && f !== null && (f === void 0 || f.$$typeof !== Ia) && !$1.has(a) && ($1.add(a), h = f === void 0 ? " However, it is set to undefined. This can be caused by a typo or by mixing up named and default imports. This can also happen due to a circular dependency, so try moving the createContext() call to a separate file." : typeof f != "object" ? " However, it is set to a " + typeof f + "." : f.$$typeof === Gd ? " Did you accidentally pass the Context.Consumer instead?" : " However, it is set to an object with keys {" + Object.keys(f).join(", ") + "}.", console.error(
          "%s defines an invalid contextType. contextType should point to the Context object returned by React.createContext().%s",
          Ge(a) || "Component",
          h
        )), typeof f == "object" && f !== null && (d = xt(f)), f = new a(i, d), t.mode & Sa) {
          oe(!0);
          try {
            f = new a(i, d);
          } finally {
            oe(!1);
          }
        }
        if (d = t.memoizedState = f.state !== null && f.state !== void 0 ? f.state : null, f.updater = r0, t.stateNode = f, f._reactInternals = t, f._reactInternalInstance = G1, typeof a.getDerivedStateFromProps == "function" && d === null && (d = Ge(a) || "Component", V1.has(d) || (V1.add(d), console.error(
          "`%s` uses `getDerivedStateFromProps` but its initial state is %s. This is not recommended. Instead, define the initial state by assigning an object to `this.state` in the constructor of `%s`. This ensures that `getDerivedStateFromProps` arguments have a consistent shape.",
          d,
          f.state === null ? "null" : "undefined",
          d
        ))), typeof a.getDerivedStateFromProps == "function" || typeof f.getSnapshotBeforeUpdate == "function") {
          var v = h = d = null;
          if (typeof f.componentWillMount == "function" && f.componentWillMount.__suppressDeprecationWarning !== !0 ? d = "componentWillMount" : typeof f.UNSAFE_componentWillMount == "function" && (d = "UNSAFE_componentWillMount"), typeof f.componentWillReceiveProps == "function" && f.componentWillReceiveProps.__suppressDeprecationWarning !== !0 ? h = "componentWillReceiveProps" : typeof f.UNSAFE_componentWillReceiveProps == "function" && (h = "UNSAFE_componentWillReceiveProps"), typeof f.componentWillUpdate == "function" && f.componentWillUpdate.__suppressDeprecationWarning !== !0 ? v = "componentWillUpdate" : typeof f.UNSAFE_componentWillUpdate == "function" && (v = "UNSAFE_componentWillUpdate"), d !== null || h !== null || v !== null) {
            f = Ge(a) || "Component";
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
        f = t.stateNode, d = Ge(a) || "Component", f.render || (a.prototype && typeof a.prototype.render == "function" ? console.error(
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
          Ge(a) || "A pure component"
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
          Ge(a)
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
        ), f = t.stateNode, f.props = i, f.state = t.memoizedState, f.refs = {}, ca(t), d = a.contextType, f.context = typeof d == "object" && d !== null ? xt(d) : nf, f.state === i && (d = Ge(a) || "Component", Z1.has(d) || (Z1.add(d), console.error(
          "%s: It is not recommended to assign props directly to state because updates to props won't be reflected in state. In most cases, it is better to use props directly.",
          d
        ))), t.mode & Sa && ku.recordLegacyContextWarning(
          t,
          f
        ), ku.recordUnsafeLifecycleWarnings(
          t,
          f
        ), f.state = t.memoizedState, d = a.getDerivedStateFromProps, typeof d == "function" && (Xt(
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
        )), yo(t, i, f, o), Bn(), f.state = t.memoizedState), typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ju) !== Yt && (t.flags |= 134217728), f = !0;
      } else if (e === null) {
        f = t.stateNode;
        var B = t.memoizedProps;
        h = bi(a, B), f.props = h;
        var L = f.context;
        v = a.contextType, d = nf, typeof v == "object" && v !== null && (d = xt(v)), b = a.getDerivedStateFromProps, v = typeof b == "function" || typeof f.getSnapshotBeforeUpdate == "function", B = t.pendingProps !== B, v || typeof f.UNSAFE_componentWillReceiveProps != "function" && typeof f.componentWillReceiveProps != "function" || (B || L !== d) && Zs(
          t,
          f,
          i,
          d
        ), uf = !1;
        var H = t.memoizedState;
        f.state = H, yo(t, i, f, o), Bn(), L = t.memoizedState, B || H !== L || uf ? (typeof b == "function" && (Xt(
          t,
          a,
          b,
          i
        ), L = t.memoizedState), (h = uf || Qs(
          t,
          a,
          h,
          i,
          H,
          L,
          d
        )) ? (v || typeof f.UNSAFE_componentWillMount != "function" && typeof f.componentWillMount != "function" || (typeof f.componentWillMount == "function" && f.componentWillMount(), typeof f.UNSAFE_componentWillMount == "function" && f.UNSAFE_componentWillMount()), typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ju) !== Yt && (t.flags |= 134217728)) : (typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ju) !== Yt && (t.flags |= 134217728), t.memoizedProps = i, t.memoizedState = L), f.props = i, f.state = L, f.context = d, f = h) : (typeof f.componentDidMount == "function" && (t.flags |= 4194308), (t.mode & Ju) !== Yt && (t.flags |= 134217728), f = !1);
      } else {
        f = t.stateNode, si(e, t), d = t.memoizedProps, v = bi(a, d), f.props = v, b = t.pendingProps, H = f.context, L = a.contextType, h = nf, typeof L == "object" && L !== null && (h = xt(L)), B = a.getDerivedStateFromProps, (L = typeof B == "function" || typeof f.getSnapshotBeforeUpdate == "function") || typeof f.UNSAFE_componentWillReceiveProps != "function" && typeof f.componentWillReceiveProps != "function" || (d !== b || H !== h) && Zs(
          t,
          f,
          i,
          h
        ), uf = !1, H = t.memoizedState, f.state = H, yo(t, i, f, o), Bn();
        var X = t.memoizedState;
        d !== b || H !== X || uf || e !== null && e.dependencies !== null && oi(e.dependencies) ? (typeof B == "function" && (Xt(
          t,
          a,
          B,
          i
        ), X = t.memoizedState), (v = uf || Qs(
          t,
          a,
          v,
          i,
          H,
          X,
          h
        ) || e !== null && e.dependencies !== null && oi(e.dependencies)) ? (L || typeof f.UNSAFE_componentWillUpdate != "function" && typeof f.componentWillUpdate != "function" || (typeof f.componentWillUpdate == "function" && f.componentWillUpdate(i, X, h), typeof f.UNSAFE_componentWillUpdate == "function" && f.UNSAFE_componentWillUpdate(
          i,
          X,
          h
        )), typeof f.componentDidUpdate == "function" && (t.flags |= 4), typeof f.getSnapshotBeforeUpdate == "function" && (t.flags |= 1024)) : (typeof f.componentDidUpdate != "function" || d === e.memoizedProps && H === e.memoizedState || (t.flags |= 4), typeof f.getSnapshotBeforeUpdate != "function" || d === e.memoizedProps && H === e.memoizedState || (t.flags |= 1024), t.memoizedProps = i, t.memoizedState = X), f.props = i, f.state = X, f.context = h, f = v) : (typeof f.componentDidUpdate != "function" || d === e.memoizedProps && H === e.memoizedState || (t.flags |= 4), typeof f.getSnapshotBeforeUpdate != "function" || d === e.memoizedProps && H === e.memoizedState || (t.flags |= 1024), f = !1);
      }
      if (h = f, ar(e, t), d = (t.flags & 128) !== 0, h || d) {
        if (h = t.stateNode, Af(t), d && typeof a.getDerivedStateFromError != "function")
          a = null, tn = -1;
        else {
          if (wt(t), a = O1(h), t.mode & Sa) {
            oe(!0);
            try {
              O1(h);
            } finally {
              oe(!1);
            }
          }
          na();
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
        )) : al(e, t, a, o), t.memoizedState = h.state, e = t.child;
      } else
        e = Zn(
          e,
          t,
          o
        );
      return o = t.stateNode, f && o.props !== i && (Sh || console.error(
        "It looks like %s is reassigning its own `this.props` while rendering. This is not supported and can lead to confusing bugs.",
        de(t) || "a component"
      ), Sh = !0), e;
    }
    function Ry(e, t, a, i) {
      return ic(), t.flags |= 256, al(e, t, a, i), t.child;
    }
    function Is(e, t) {
      t && t.childContextTypes && console.error(
        `childContextTypes cannot be defined on a function component.
  %s.childContextTypes = ...`,
        t.displayName || t.name || "Component"
      ), typeof t.getDerivedStateFromProps == "function" && (e = Ge(t) || "Unknown", tb[e] || (console.error(
        "%s: Function components do not support getDerivedStateFromProps.",
        e
      ), tb[e] = !0)), typeof t.contextType == "object" && t.contextType !== null && (t = Ge(t) || "Unknown", P1[t] || (console.error(
        "%s: Function components do not support contextType.",
        t
      ), P1[t] = !0));
    }
    function nr(e) {
      return { baseLanes: e, cachePool: Jp() };
    }
    function Ps(e, t, a) {
      return e = e !== null ? e.childLanes & ~a : 0, t && (e |= On), e;
    }
    function Ip(e, t, a) {
      var i, o = t.pendingProps;
      he(t) && (t.flags |= 128);
      var f = !1, d = (t.flags & 128) !== 0;
      if ((i = d) || (i = e !== null && e.memoizedState === null ? !1 : (jl.current & np) !== 0), i && (f = !0, t.flags &= -129), i = (t.flags & 32) !== 0, t.flags &= -33, e === null) {
        if (mt) {
          if (f ? Da(t) : vn(t), mt) {
            var h = nl, v;
            if (!(v = !h)) {
              e: {
                var b = h;
                for (v = ji; b.nodeType !== 8; ) {
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
                treeContext: Vr !== null ? { id: Gc, overflow: Lc } : null,
                retryLane: 536870912,
                hydrationErrors: null
              }, b = U(18, null, null, Yt), b.stateNode = v, b.return = t, t.child = b, Na = t, nl = null, v = !0) : v = !1, v = !v;
            }
            v && (Ih(
              t,
              h
            ), Hn(t));
          }
          if (h = t.memoizedState, h !== null && (h = h.dehydrated, h !== null))
            return lu(h) ? t.lanes = 32 : t.lanes = 536870912, null;
          za(t);
        }
        return h = o.children, o = o.fallback, f ? (vn(t), f = t.mode, h = ur(
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
        ), h.return = t, o.return = t, h.sibling = o, t.child = h, f = t.child, f.memoizedState = nr(a), f.childLanes = Ps(
          e,
          i,
          a
        ), t.memoizedState = y0, o) : (Da(t), ed(
          t,
          h
        ));
      }
      var B = e.memoizedState;
      if (B !== null && (h = B.dehydrated, h !== null)) {
        if (d)
          t.flags & 256 ? (Da(t), t.flags &= -257, t = td(
            e,
            t,
            a
          )) : t.memoizedState !== null ? (vn(t), t.child = e.child, t.flags |= 128, t = null) : (vn(t), f = o.fallback, h = t.mode, o = ur(
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
        else if (Da(t), mt && console.error(
          "We should not be hydrating here. This is a bug in React. Please file a bug."
        ), lu(h)) {
          if (i = h.nextSibling && h.nextSibling.dataset, i) {
            v = i.dgst;
            var L = i.msg;
            b = i.stck;
            var H = i.cstck;
          }
          h = L, i = v, o = b, v = f = H, f = Error(h || "The server could not finish this Suspense boundary, likely due to an error during server rendering. Switched to client rendering."), f.stack = o || "", f.digest = i, i = v === void 0 ? null : v, o = {
            value: f,
            source: null,
            stack: i
          }, typeof i == "string" && kg.set(
            f,
            o
          ), ro(o), t = td(
            e,
            t,
            a
          );
        } else if (Kl || _l(
          e,
          t,
          a,
          !1
        ), i = (a & e.childLanes) !== 0, Kl || i) {
          if (i = Ht, i !== null && (o = a & -a, o = (o & 42) !== 0 ? 1 : Ol(
            o
          ), o = (o & (i.suspendedLanes | a)) !== 0 ? 0 : o, o !== 0 && o !== B.retryLane))
            throw B.retryLane = o, ia(
              e,
              o
            ), Kt(
              i,
              e,
              o
            ), F1;
          h.data === Jc || hd(), t = td(
            e,
            t,
            a
          );
        } else
          h.data === Jc ? (t.flags |= 192, t.child = e.child, t = null) : (e = B.treeContext, nl = Nl(
            h.nextSibling
          ), Na = t, mt = !0, Xr = null, Vc = !1, du = null, ji = !1, e !== null && (fn(), ru[su++] = Gc, ru[su++] = Lc, ru[su++] = Vr, Gc = e.id, Lc = e.overflow, Vr = t), t = ed(
            t,
            o.children
          ), t.flags |= 4096);
        return t;
      }
      return f ? (vn(t), f = o.fallback, h = t.mode, v = e.child, b = v.sibling, o = xn(
        v,
        {
          mode: "hidden",
          children: o.children
        }
      ), o.subtreeFlags = v.subtreeFlags & 65011712, b !== null ? f = xn(
        b,
        f
      ) : (f = ui(
        f,
        h,
        a,
        null
      ), f.flags |= 2), f.return = t, o.return = t, o.sibling = f, t.child = o, o = f, f = t.child, h = e.child.memoizedState, h === null ? h = nr(a) : (v = h.cachePool, v !== null ? (b = Bl._currentValue, v = v.parent !== b ? { parent: b, pool: b } : v) : v = Jp(), h = {
        baseLanes: h.baseLanes | a,
        cachePool: v
      }), f.memoizedState = h, f.childLanes = Ps(
        e,
        i,
        a
      ), t.memoizedState = y0, o) : (Da(t), a = e.child, e = a.sibling, a = xn(a, {
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
      return e = U(22, e, null, t), e.lanes = 0, e.stateNode = {
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
      return e = !a && typeof bt(e) == "function", a || e ? (a = a ? "array" : "iterable", console.error(
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
        } else if (d = bt(i), typeof d == "function") {
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
      if (al(e, t, i, a), i = jl.current, (i & np) !== 0)
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
      switch (Ce(jl, i, t), o) {
        case "forwards":
          for (a = t.child, o = null; a !== null; )
            e = a.alternate, e !== null && xu(e) === null && (o = a), a = a.sibling;
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
            if (e = o.alternate, e !== null && xu(e) === null) {
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
    function Zn(e, t, a) {
      if (e !== null && (t.dependencies = e.dependencies), tn = -1, rf |= t.lanes, (a & t.childLanes) === 0)
        if (e !== null) {
          if (_l(
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
        for (e = t.child, a = xn(e, e.pendingProps), t.child = a, a.return = t; e.sibling !== null; )
          e = e.sibling, a = a.sibling = xn(e, e.pendingProps), a.return = t;
        a.sibling = null;
      }
      return t.child;
    }
    function nd(e, t) {
      return (e.lanes & t) !== 0 ? !0 : (e = e.dependencies, !!(e !== null && oi(e)));
    }
    function Ug(e, t, a) {
      switch (t.tag) {
        case 3:
          Gt(
            t,
            t.stateNode.containerInfo
          ), ci(
            t,
            Bl,
            e.memoizedState.cache
          ), ic();
          break;
        case 27:
        case 5:
          W(t);
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
            return i.dehydrated !== null ? (Da(t), t.flags |= 128, null) : (a & t.child.childLanes) !== 0 ? Ip(
              e,
              t,
              a
            ) : (Da(t), e = Zn(
              e,
              t,
              a
            ), e !== null ? e.sibling : null);
          Da(t);
          break;
        case 19:
          var o = (e.flags & 128) !== 0;
          if (i = (a & t.childLanes) !== 0, i || (_l(
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
          if (o = t.memoizedState, o !== null && (o.rendering = null, o.tail = null, o.lastEffect = null), Ce(
            jl,
            jl.current,
            t
          ), i) break;
          return null;
        case 22:
        case 23:
          return t.lanes = 0, $s(e, t, a);
        case 24:
          ci(
            t,
            Bl,
            e.memoizedState.cache
          );
      }
      return Zn(e, t, a);
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
          Kl = !0;
        else {
          if (!nd(e, a) && (t.flags & 128) === 0)
            return Kl = !1, Ug(
              e,
              t,
              a
            );
          Kl = (e.flags & 131072) !== 0;
        }
      else
        Kl = !1, (i = mt) && (fn(), i = (t.flags & 1048576) !== 0), i && (i = t.index, fn(), Qp(t, Bv, i));
      switch (t.lanes = 0, t.tag) {
        case 16:
          e: if (i = t.pendingProps, e = of(t.elementType), t.type = e, typeof e == "function")
            $h(e) ? (i = bi(
              e,
              i
            ), t.tag = 1, t.type = e = ac(e), t = Ay(
              null,
              t,
              e,
              i,
              a
            )) : (t.tag = 0, Is(t, e), t.type = e = ac(e), t = Fs(
              null,
              t,
              e,
              i,
              a
            ));
          else {
            if (e != null) {
              if (o = e.$$typeof, o === Gu) {
                t.tag = 11, t.type = e = kh(e), t = ks(
                  null,
                  t,
                  e,
                  i,
                  a
                );
                break e;
              } else if (o === Ur) {
                t.tag = 14, t = Qn(
                  null,
                  t,
                  e,
                  i,
                  a
                );
                break e;
              }
            }
            throw t = "", e !== null && typeof e == "object" && e.$$typeof === Ca && (t = " Did you wrap a component in React.lazy() more than once?"), e = Ge(e) || e, Error(
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
          return i = t.type, o = bi(
            i,
            t.pendingProps
          ), Ay(
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
            o = f.element, si(e, t), yo(t, i, null, a);
            var d = t.memoizedState;
            if (i = d.cache, ci(t, Bl, i), i !== f.cache && ny(
              t,
              [Bl],
              a,
              !0
            ), Bn(), i = d.element, f.isDehydrated)
              if (f = {
                element: i,
                isDehydrated: !1,
                cache: d.cache
              }, t.updateQueue.baseState = f, t.memoizedState = f, t.flags & 256) {
                t = Ry(
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
                ), ro(o), t = Ry(
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
                for (nl = Nl(e.firstChild), Na = t, mt = !0, Xr = null, Vc = !1, du = null, ji = !0, e = j1(
                  t,
                  null,
                  i,
                  a
                ), t.child = e; e; )
                  e.flags = e.flags & -3 | 4096, e = e.sibling;
              }
            else {
              if (ic(), i === o) {
                t = Zn(
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
          return ar(e, t), e === null ? (e = ju(
            t.type,
            null,
            t.pendingProps,
            null
          )) ? t.memoizedState = e : mt || (e = t.type, a = t.pendingProps, i = _t(
            nu.current
          ), i = nt(
            i
          ).createElement(e), i[Zl] = t, i[ga] = a, kt(i, e, a), D(i), t.stateNode = i) : t.memoizedState = ju(
            t.type,
            e.memoizedProps,
            t.pendingProps,
            e.memoizedState
          ), null;
        case 27:
          return W(t), e === null && mt && (i = _t(nu.current), o = O(), i = t.stateNode = sm(
            t.type,
            t.pendingProps,
            i,
            o,
            !1
          ), Vc || (o = Ut(
            i,
            t.type,
            t.pendingProps,
            o
          ), o !== null && (rn(t, 0).serverProps = o)), Na = t, ji = !0, o = nl, tu(t.type) ? (B0 = o, nl = Nl(
            i.firstChild
          )) : nl = o), al(
            e,
            t,
            t.pendingProps.children,
            a
          ), ar(e, t), e === null && (t.flags |= 4194304), t.child;
        case 5:
          return e === null && mt && (f = O(), i = ps(
            t.type,
            f.ancestorInfo
          ), o = nl, (d = !o) || (d = Di(
            o,
            t.type,
            t.pendingProps,
            ji
          ), d !== null ? (t.stateNode = d, Vc || (f = Ut(
            d,
            t.type,
            t.pendingProps,
            f
          ), f !== null && (rn(t, 0).serverProps = f)), Na = t, nl = Nl(
            d.firstChild
          ), ji = !1, f = !0) : f = !1, d = !f), d && (i && Ih(t, o), Hn(t))), W(t), o = t.type, f = t.pendingProps, d = e !== null ? e.memoizedProps : null, i = f.children, eu(o, f) ? i = null : d !== null && eu(o, d) && (t.flags |= 32), t.memoizedState !== null && (o = yi(
            e,
            t,
            Xa,
            null,
            null,
            a
          ), bp._currentValue = o), ar(e, t), al(
            e,
            t,
            i,
            a
          ), t.child;
        case 6:
          return e === null && mt && (e = t.pendingProps, a = O(), i = a.ancestorInfo.current, e = i != null ? Uf(
            e,
            i.tag,
            a.ancestorInfo.implicitRootScope
          ) : !0, a = nl, (i = !a) || (i = Hl(
            a,
            t.pendingProps,
            ji
          ), i !== null ? (t.stateNode = i, Na = t, nl = null, i = !0) : i = !1, i = !i), i && (e && Ih(t, a), Hn(t))), null;
        case 13:
          return Ip(e, t, a);
        case 4:
          return Gt(
            t,
            t.stateNode.containerInfo
          ), i = t.pendingProps, e === null ? t.child = vh(
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
          return ks(
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
          return i = t.type, o = t.pendingProps, f = o.value, "value" in o || ab || (ab = !0, console.error(
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
          ), fi(t), o = xt(o), wt(t), i = c0(
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
          return Qn(
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
          ), e.ref = t.ref, t.child = e, e.return = t, t = e) : (e = xn(e.child, i), e.ref = t.ref, t.child = e, e.return = t, t = e), t;
        case 22:
          return $s(e, t, a);
        case 24:
          return fi(t), i = xt(Bl), e === null ? (o = cy(), o === null && (o = Ht, f = jf(), o.pooledCache = f, cc(f), f !== null && (o.pooledCacheLanes |= a), o = f), t.memoizedState = {
            parent: i,
            cache: o
          }, ca(t), ci(t, Bl, o)) : ((e.lanes & a) !== 0 && (si(e, t), yo(t, null, null, a), Bn()), o = e.memoizedState, f = t.memoizedState, o.parent !== i ? (o = {
            parent: i,
            cache: i
          }, t.memoizedState = o, t.lanes === 0 && (t.memoizedState = t.updateQueue.baseState = o), ci(t, Bl, i)) : (i = f.cache, ci(t, Bl, i), i !== o.cache && ny(
            t,
            [Bl],
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
    function sa(e) {
      e.flags |= 4;
    }
    function ir(e, t) {
      if (t.type !== "stylesheet" || (t.state.loading & vu) !== ns)
        e.flags &= -16777217;
      else if (e.flags |= 16777216, !Sr(t)) {
        if (t = pu.current, t !== null && ((it & 4194048) === it ? Li !== null : (it & 62914560) !== it && (it & 536870912) === 0 || t !== Li))
          throw ep = e0, h1;
        e.flags |= 8192;
      }
    }
    function cr(e, t) {
      t !== null && (e.flags |= 4), e.flags & 16384 && (t = e.tag !== 22 ? Un() : 536870912, e.lanes |= t, Ir |= t);
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
    function Ot(e) {
      var t = e.alternate !== null && e.alternate.child === e.child, a = 0, i = 0;
      if (t)
        if ((e.mode & ta) !== Yt) {
          for (var o = e.selfBaseDuration, f = e.child; f !== null; )
            a |= f.lanes | f.childLanes, i |= f.subtreeFlags & 65011712, i |= f.flags & 65011712, o += f.treeBaseDuration, f = f.sibling;
          e.treeBaseDuration = o;
        } else
          for (o = e.child; o !== null; )
            a |= o.lanes | o.childLanes, i |= o.subtreeFlags & 65011712, i |= o.flags & 65011712, o.return = e, o = o.sibling;
      else if ((e.mode & ta) !== Yt) {
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
          return Ot(t), null;
        case 1:
          return Ot(t), null;
        case 3:
          return a = t.stateNode, i = null, e !== null && (i = e.memoizedState.cache), t.memoizedState.cache !== i && (t.flags |= 2048), Ou(Bl, t), pt(t), a.pendingContext && (a.context = a.pendingContext, a.pendingContext = null), (e === null || e.child === null) && (uc(t) ? (ly(), sa(t)) : e === null || e.memoizedState.isDehydrated && (t.flags & 256) === 0 || (t.flags |= 1024, ty())), Ot(t), null;
        case 26:
          return a = t.memoizedState, e === null ? (sa(t), a !== null ? (Ot(t), ir(
            t,
            a
          )) : (Ot(t), t.flags &= -16777217)) : a ? a !== e.memoizedState ? (sa(t), Ot(t), ir(
            t,
            a
          )) : (Ot(t), t.flags &= -16777217) : (e.memoizedProps !== i && sa(t), Ot(t), t.flags &= -16777217), null;
        case 27:
          P(t), a = _t(nu.current);
          var o = t.type;
          if (e !== null && t.stateNode != null)
            e.memoizedProps !== i && sa(t);
          else {
            if (!i) {
              if (t.stateNode === null)
                throw Error(
                  "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
                );
              return Ot(t), null;
            }
            e = O(), uc(t) ? Ph(t) : (e = sm(
              o,
              i,
              a,
              e,
              !0
            ), t.stateNode = e, sa(t));
          }
          return Ot(t), null;
        case 5:
          if (P(t), a = t.type, e !== null && t.stateNode != null)
            e.memoizedProps !== i && sa(t);
          else {
            if (!i) {
              if (t.stateNode === null)
                throw Error(
                  "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
                );
              return Ot(t), null;
            }
            if (o = O(), uc(t))
              Ph(t);
            else {
              switch (e = _t(nu.current), ps(a, o.ancestorInfo), o = o.context, e = nt(e), o) {
                case Uh:
                  e = e.createElementNS(af, a);
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
                        af,
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
                      ), Object.prototype.toString.call(e) !== "[object HTMLUnknownElement]" || Vu.call(
                        Eb,
                        a
                      ) || (Eb[a] = !0, console.error(
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
              e && sa(t);
            }
          }
          return Ot(t), t.flags &= -16777217, null;
        case 6:
          if (e && t.stateNode != null)
            e.memoizedProps !== i && sa(t);
          else {
            if (typeof i != "string" && t.stateNode === null)
              throw Error(
                "We must have new props for new mounts. This error is likely caused by a bug in React. Please file an issue."
              );
            if (e = _t(nu.current), a = O(), uc(t)) {
              e = t.stateNode, a = t.memoizedProps, o = !Vc, i = null;
              var f = Na;
              if (f !== null)
                switch (f.tag) {
                  case 3:
                    o && (o = Cd(
                      e,
                      a,
                      i
                    ), o !== null && (rn(t, 0).serverProps = o));
                    break;
                  case 27:
                  case 5:
                    i = f.memoizedProps, o && (o = Cd(
                      e,
                      a,
                      i
                    ), o !== null && (rn(
                      t,
                      0
                    ).serverProps = o));
                }
              e[Zl] = t, e = !!(e.nodeValue === a || i !== null && i.suppressHydrationWarning === !0 || tm(e.nodeValue, a)), e || Hn(t);
            } else
              o = a.ancestorInfo.current, o != null && Uf(
                i,
                o.tag,
                a.ancestorInfo.implicitRootScope
              ), e = nt(e).createTextNode(
                i
              ), e[Zl] = t, t.stateNode = e;
          }
          return Ot(t), null;
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
                o[Zl] = t, Ot(t), (t.mode & ta) !== Yt && i !== null && (o = t.child, o !== null && (t.treeBaseDuration -= o.treeBaseDuration));
              } else
                ly(), ic(), (t.flags & 128) === 0 && (t.memoizedState = null), t.flags |= 4, Ot(t), (t.mode & ta) !== Yt && i !== null && (o = t.child, o !== null && (t.treeBaseDuration -= o.treeBaseDuration));
              o = !1;
            } else
              o = ty(), e !== null && e.memoizedState !== null && (e.memoizedState.hydrationErrors = o), o = !0;
            if (!o)
              return t.flags & 256 ? (za(t), t) : (za(t), null);
          }
          return za(t), (t.flags & 128) !== 0 ? (t.lanes = a, (t.mode & ta) !== Yt && wn(t), t) : (a = i !== null, e = e !== null && e.memoizedState !== null, a && (i = t.child, o = null, i.alternate !== null && i.alternate.memoizedState !== null && i.alternate.memoizedState.cachePool !== null && (o = i.alternate.memoizedState.cachePool.pool), f = null, i.memoizedState !== null && i.memoizedState.cachePool !== null && (f = i.memoizedState.cachePool.pool), f !== o && (i.flags |= 2048)), a !== e && a && (t.child.flags |= 8192), cr(t, t.updateQueue), Ot(t), (t.mode & ta) !== Yt && a && (e = t.child, e !== null && (t.treeBaseDuration -= e.treeBaseDuration)), null);
        case 4:
          return pt(t), e === null && Py(
            t.stateNode.containerInfo
          ), Ot(t), null;
        case 10:
          return Ou(t.type, t), Ot(t), null;
        case 19:
          if (Te(jl, t), o = t.memoizedState, o === null) return Ot(t), null;
          if (i = (t.flags & 128) !== 0, f = o.rendering, f === null)
            if (i) Si(o, !1);
            else {
              if (ul !== Kc || e !== null && (e.flags & 128) !== 0)
                for (e = t.child; e !== null; ) {
                  if (f = xu(e), f !== null) {
                    for (t.flags |= 128, Si(o, !1), e = f.updateQueue, t.updateQueue = e, cr(t, e), t.subtreeFlags = 0, e = a, a = t.child; a !== null; )
                      Wh(a, e), a = a.sibling;
                    return Ce(
                      jl,
                      jl.current & gh | np,
                      t
                    ), t.child;
                  }
                  e = e.sibling;
                }
              o.tail !== null && uu() > Iv && (t.flags |= 128, i = !0, Si(o, !1), t.lanes = 4194304);
            }
          else {
            if (!i)
              if (e = xu(f), e !== null) {
                if (t.flags |= 128, i = !0, e = e.updateQueue, t.updateQueue = e, cr(t, e), Si(o, !0), o.tail === null && o.tailMode === "hidden" && !f.alternate && !mt)
                  return Ot(t), null;
              } else
                2 * uu() - o.renderingStartTime > Iv && a !== 536870912 && (t.flags |= 128, i = !0, Si(o, !1), t.lanes = 4194304);
            o.isBackwards ? (f.sibling = t.child, t.child = f) : (e = o.last, e !== null ? e.sibling = f : t.child = f, o.last = f);
          }
          return o.tail !== null ? (e = o.tail, o.rendering = e, o.tail = e.sibling, o.renderingStartTime = uu(), e.sibling = null, a = jl.current, a = i ? a & gh | np : a & gh, Ce(jl, a, t), e) : (Ot(t), null);
        case 22:
        case 23:
          return za(t), yn(t), i = t.memoizedState !== null, e !== null ? e.memoizedState !== null !== i && (t.flags |= 8192) : i && (t.flags |= 8192), i ? (a & 536870912) !== 0 && (t.flags & 128) === 0 && (Ot(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : Ot(t), a = t.updateQueue, a !== null && cr(t, a.retryQueue), a = null, e !== null && e.memoizedState !== null && e.memoizedState.cachePool !== null && (a = e.memoizedState.cachePool.pool), i = null, t.memoizedState !== null && t.memoizedState.cachePool !== null && (i = t.memoizedState.cachePool.pool), i !== a && (t.flags |= 2048), e !== null && Te(Kr, t), null;
        case 24:
          return a = null, e !== null && (a = e.memoizedState.cache), t.memoizedState.cache !== a && (t.flags |= 2048), Ou(Bl, t), Ot(t), null;
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
          return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & ta) !== Yt && wn(t), t) : null;
        case 3:
          return Ou(Bl, t), pt(t), e = t.flags, (e & 65536) !== 0 && (e & 128) === 0 ? (t.flags = e & -65537 | 128, t) : null;
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
          return e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & ta) !== Yt && wn(t), t) : null;
        case 19:
          return Te(jl, t), null;
        case 4:
          return pt(t), null;
        case 10:
          return Ou(t.type, t), null;
        case 22:
        case 23:
          return za(t), yn(t), e !== null && Te(Kr, t), e = t.flags, e & 65536 ? (t.flags = e & -65537 | 128, (t.mode & ta) !== Yt && wn(t), t) : null;
        case 24:
          return Ou(Bl, t), null;
        case 25:
          return null;
        default:
          return null;
      }
    }
    function zy(e, t) {
      switch (zs(t), t.tag) {
        case 3:
          Ou(Bl, t), pt(t);
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
          Te(jl, t);
          break;
        case 10:
          Ou(t.type, t);
          break;
        case 22:
        case 23:
          za(t), yn(t), e !== null && Te(Kr, t);
          break;
        case 24:
          Ou(Bl, t);
      }
    }
    function gn(e) {
      return (e.mode & ta) !== Yt;
    }
    function My(e, t) {
      gn(e) ? (dn(), pc(t, e), Ga()) : pc(t, e);
    }
    function id(e, t, a) {
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
            if ((a.tag & e) === e && ((e & Yl) !== hu ? fe !== null && typeof fe.markComponentPassiveEffectMountStarted == "function" && fe.markComponentPassiveEffectMountStarted(
              t
            ) : (e & la) !== hu && fe !== null && typeof fe.markComponentLayoutEffectMountStarted == "function" && fe.markComponentLayoutEffectMountStarted(
              t
            ), i = void 0, (e & wa) !== hu && (zh = !0), i = ye(
              t,
              QS,
              a
            ), (e & wa) !== hu && (zh = !1), (e & Yl) !== hu ? fe !== null && typeof fe.markComponentPassiveEffectMountStopped == "function" && fe.markComponentPassiveEffectMountStopped() : (e & la) !== hu && fe !== null && typeof fe.markComponentLayoutEffectMountStopped == "function" && fe.markComponentLayoutEffectMountStopped(), i !== void 0 && typeof i != "function")) {
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
        Ue(t, t.return, h);
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
              h !== void 0 && (d.destroy = void 0, (e & Yl) !== hu ? fe !== null && typeof fe.markComponentPassiveEffectUnmountStarted == "function" && fe.markComponentPassiveEffectUnmountStarted(
                t
              ) : (e & la) !== hu && fe !== null && typeof fe.markComponentLayoutEffectUnmountStarted == "function" && fe.markComponentLayoutEffectUnmountStarted(
                t
              ), (e & wa) !== hu && (zh = !0), o = t, ye(
                o,
                ZS,
                o,
                a,
                h
              ), (e & wa) !== hu && (zh = !1), (e & Yl) !== hu ? fe !== null && typeof fe.markComponentPassiveEffectUnmountStopped == "function" && fe.markComponentPassiveEffectUnmountStopped() : (e & la) !== hu && fe !== null && typeof fe.markComponentLayoutEffectUnmountStopped == "function" && fe.markComponentLayoutEffectUnmountStopped());
            }
            i = i.next;
          } while (i !== f);
        }
      } catch (v) {
        Ue(t, t.return, v);
      }
    }
    function Uy(e, t) {
      gn(e) ? (dn(), pc(t, e), Ga()) : pc(t, e);
    }
    function or(e, t, a) {
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
        e.type.defaultProps || "ref" in e.memoizedProps || Sh || (a.props !== e.memoizedProps && console.error(
          "Expected %s props to match memoized props before processing the update queue. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
          de(e) || "instance"
        ), a.state !== e.memoizedState && console.error(
          "Expected %s state to match memoized state before processing the update queue. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
          de(e) || "instance"
        ));
        try {
          ye(
            e,
            kp,
            t,
            a
          );
        } catch (i) {
          Ue(e, e.return, i);
        }
      }
    }
    function tv(e, t, a) {
      return e.getSnapshotBeforeUpdate(t, a);
    }
    function _g(e, t) {
      var a = t.memoizedProps, i = t.memoizedState;
      t = e.stateNode, e.type.defaultProps || "ref" in e.memoizedProps || Sh || (t.props !== e.memoizedProps && console.error(
        "Expected %s props to match memoized props before getSnapshotBeforeUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
        de(e) || "instance"
      ), t.state !== e.memoizedState && console.error(
        "Expected %s state to match memoized state before getSnapshotBeforeUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
        de(e) || "instance"
      ));
      try {
        var o = bi(
          e.type,
          a,
          e.elementType === e.type
        ), f = ye(
          e,
          tv,
          t,
          o,
          i
        );
        a = nb, f !== void 0 || a.has(e.type) || (a.add(e.type), ye(e, function() {
          console.error(
            "%s.getSnapshotBeforeUpdate(): A snapshot value (or null) must be returned. You have returned undefined.",
            de(e)
          );
        })), t.__reactInternalSnapshotBeforeUpdate = f;
      } catch (d) {
        Ue(e, e.return, d);
      }
    }
    function cd(e, t, a) {
      a.props = bi(
        e.type,
        e.memoizedProps
      ), a.state = e.memoizedState, gn(e) ? (dn(), ye(
        e,
        C1,
        e,
        t,
        a
      ), Ga()) : ye(
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
            de(e)
          ), t.current = a;
      }
    }
    function Mo(e, t) {
      try {
        ye(e, lv, e);
      } catch (a) {
        Ue(e, t, a);
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
            Ue(e, t, o);
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
            Ue(e, t, o);
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
        ye(
          e,
          Bu,
          i,
          t,
          a,
          e
        );
      } catch (o) {
        Ue(e, e.return, o);
      }
    }
    function xy(e, t, a) {
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
        Ue(e, e.return, i);
      }
    }
    function Hy(e) {
      return e.tag === 5 || e.tag === 3 || e.tag === 26 || e.tag === 27 && tu(e.type) || e.tag === 4;
    }
    function gc(e) {
      e: for (; ; ) {
        for (; e.sibling === null; ) {
          if (e.return === null || Hy(e.return)) return null;
          e = e.return;
        }
        for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
          if (e.tag === 27 && tu(e.type) || e.flags & 2 || e.child === null || e.tag === 4) continue e;
          e.child.return = e, e = e.child;
        }
        if (!(e.flags & 2)) return e.stateNode;
      }
    }
    function fr(e, t, a) {
      var i = e.tag;
      if (i === 5 || i === 6)
        e = e.stateNode, t ? (a.nodeType === 9 ? a.body : a.nodeName === "HTML" ? a.ownerDocument.body : a).insertBefore(e, t) : (t = a.nodeType === 9 ? a.body : a.nodeName === "HTML" ? a.ownerDocument.body : a, t.appendChild(e), a = a._reactRootContainer, a != null || t.onclick !== null || (t.onclick = qu));
      else if (i !== 4 && (i === 27 && tu(e.type) && (a = e.stateNode, t = null), e = e.child, e !== null))
        for (fr(e, t, a), e = e.sibling; e !== null; )
          fr(e, t, a), e = e.sibling;
    }
    function bc(e, t, a) {
      var i = e.tag;
      if (i === 5 || i === 6)
        e = e.stateNode, t ? a.insertBefore(e, t) : a.appendChild(e);
      else if (i !== 4 && (i === 27 && tu(e.type) && (a = e.stateNode), e = e.child, e !== null))
        for (bc(e, t, a), e = e.sibling; e !== null; )
          bc(e, t, a), e = e.sibling;
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
          t = t.stateNode, a = gc(e), bc(
            e,
            a,
            t
          );
          break;
        case 5:
          a = t.stateNode, t.flags & 32 && (Yu(a), t.flags &= -33), t = gc(e), bc(
            e,
            t,
            a
          );
          break;
        case 3:
        case 4:
          t = t.stateNode.containerInfo, a = gc(e), fr(
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
        ye(
          e,
          _a,
          e.type,
          a,
          t,
          e
        );
      } catch (i) {
        Ue(e, e.return, i);
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
              var d = 0, h = -1, v = -1, b = 0, B = 0, L = e, H = null;
              t: for (; ; ) {
                for (var X; L !== a || o !== 0 && L.nodeType !== 3 || (h = d + o), L !== f || i !== 0 && L.nodeType !== 3 || (v = d + i), L.nodeType === 3 && (d += L.nodeValue.length), (X = L.firstChild) !== null; )
                  H = L, L = X;
                for (; ; ) {
                  if (L === e) break t;
                  if (H === a && ++b === o && (h = d), H === f && ++B === i && (v = d), (X = L.nextSibling) !== null) break;
                  L = H, H = L.parentNode;
                }
                L = X;
              }
              a = h === -1 || v === -1 ? null : { start: h, end: v };
            } else a = null;
          }
        a = a || { start: 0, end: 0 };
      } else a = null;
      for (w0 = {
        focusedElem: e,
        selectionRange: a
      }, yg = !1, Jl = t; Jl !== null; )
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
                    jo(e);
                  else if (a === 1)
                    switch (e.nodeName) {
                      case "HEAD":
                      case "HTML":
                      case "BODY":
                        jo(e);
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
    function wy(e, t, a) {
      var i = a.flags;
      switch (a.tag) {
        case 0:
        case 11:
        case 15:
          Kn(e, a), i & 4 && My(a, la | yu);
          break;
        case 1:
          if (Kn(e, a), i & 4)
            if (e = a.stateNode, t === null)
              a.type.defaultProps || "ref" in a.memoizedProps || Sh || (e.props !== a.memoizedProps && console.error(
                "Expected %s props to match memoized props before componentDidMount. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
                de(a) || "instance"
              ), e.state !== a.memoizedState && console.error(
                "Expected %s state to match memoized state before componentDidMount. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
                de(a) || "instance"
              )), gn(a) ? (dn(), ye(
                a,
                o0,
                a,
                e
              ), Ga()) : ye(
                a,
                o0,
                a,
                e
              );
            else {
              var o = bi(
                a.type,
                t.memoizedProps
              );
              t = t.memoizedState, a.type.defaultProps || "ref" in a.memoizedProps || Sh || (e.props !== a.memoizedProps && console.error(
                "Expected %s props to match memoized props before componentDidUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.props`. Please file an issue.",
                de(a) || "instance"
              ), e.state !== a.memoizedState && console.error(
                "Expected %s state to match memoized state before componentDidUpdate. This might either be because of a bug in React, or because a component reassigns its own `this.state`. Please file an issue.",
                de(a) || "instance"
              )), gn(a) ? (dn(), ye(
                a,
                M1,
                a,
                e,
                o,
                t,
                e.__reactInternalSnapshotBeforeUpdate
              ), Ga()) : ye(
                a,
                M1,
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
          if (t = sn(), Kn(e, a), i & 64 && (i = a.updateQueue, i !== null)) {
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
                kp,
                i,
                o
              );
            } catch (d) {
              Ue(a, a.return, d);
            }
          }
          e.effectDuration += ri(t);
          break;
        case 27:
          t === null && i & 4 && Ny(a);
        case 26:
        case 5:
          Kn(e, a), t === null && i & 4 && nv(a), i & 512 && Mo(a, a.return);
          break;
        case 12:
          if (i & 4) {
            i = sn(), Kn(e, a), e = a.stateNode, e.effectDuration += oc(i);
            try {
              ye(
                a,
                Cy,
                a,
                t,
                jv,
                e.effectDuration
              );
            } catch (d) {
              Ue(a, a.return, d);
            }
          } else Kn(e, a);
          break;
        case 13:
          Kn(e, a), i & 4 && Uo(e, a), i & 64 && (e = a.memoizedState, e !== null && (e = e.dehydrated, e !== null && (a = vr.bind(
            null,
            a
          ), Go(e, a))));
          break;
        case 22:
          if (i = a.memoizedState !== null || Zc, !i) {
            t = t !== null && t.memoizedState !== null || hl, o = Zc;
            var f = hl;
            Zc = i, (hl = t) && !f ? Jn(
              e,
              a,
              (a.subtreeFlags & 8772) !== 0
            ) : Kn(e, a), Zc = o, hl = f;
          }
          break;
        case 30:
          break;
        default:
          Kn(e, a);
      }
    }
    function qy(e) {
      var t = e.alternate;
      t !== null && (e.alternate = null, qy(t)), e.child = null, e.deletions = null, e.sibling = null, e.tag === 5 && (t = e.stateNode, t !== null && nn(t)), e.stateNode = null, e._debugOwner = null, e.return = null, e.dependencies = null, e.memoizedProps = null, e.memoizedState = null, e.pendingProps = null, e.stateNode = null, e.updateQueue = null;
    }
    function Hu(e, t, a) {
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
          hl || ka(a, t), Hu(
            e,
            t,
            a
          ), a.memoizedState ? a.memoizedState.count-- : a.stateNode && (a = a.stateNode, a.parentNode.removeChild(a));
          break;
        case 27:
          hl || ka(a, t);
          var i = Rl, o = ln;
          tu(a.type) && (Rl = a.stateNode, ln = !1), Hu(
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
          if (i = Rl, o = ln, Rl = null, Hu(
            e,
            t,
            a
          ), Rl = i, ln = o, Rl !== null)
            if (ln)
              try {
                ye(
                  a,
                  Bo,
                  Rl,
                  a.stateNode
                );
              } catch (f) {
                Ue(
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
                Ue(
                  a,
                  t,
                  f
                );
              }
          break;
        case 18:
          Rl !== null && (ln ? (e = Rl, Yo(
            e.nodeType === 9 ? e.body : e.nodeName === "HTML" ? e.ownerDocument.body : e,
            a.stateNode
          ), Hc(e)) : Yo(Rl, a.stateNode));
          break;
        case 4:
          i = Rl, o = ln, Rl = a.stateNode.containerInfo, ln = !0, Hu(
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
          ), hl || id(
            a,
            t,
            la
          ), Hu(
            e,
            t,
            a
          );
          break;
        case 1:
          hl || (ka(a, t), i = a.stateNode, typeof i.componentWillUnmount == "function" && cd(
            a,
            t,
            i
          )), Hu(
            e,
            t,
            a
          );
          break;
        case 21:
          Hu(
            e,
            t,
            a
          );
          break;
        case 22:
          hl = (i = hl) || a.memoizedState !== null, Hu(
            e,
            t,
            a
          ), hl = i;
          break;
        default:
          Hu(
            e,
            t,
            a
          );
      }
    }
    function Uo(e, t) {
      if (t.memoizedState === null && (e = t.alternate, e !== null && (e = e.memoizedState, e !== null && (e = e.dehydrated, e !== null))))
        try {
          ye(
            t,
            Ua,
            e
          );
        } catch (a) {
          Ue(t, t.return, a);
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
    function Tc(e, t) {
      var a = fd(e);
      t.forEach(function(i) {
        var o = Ri.bind(null, e, i);
        if (!a.has(i)) {
          if (a.add(i), Ft)
            if (Th !== null && Eh !== null)
              wo(Eh, Th);
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
                if (tu(h.type)) {
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
          By(t, e), t = t.sibling;
    }
    function By(e, t) {
      var a = e.alternate, i = e.flags;
      switch (e.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          Vl(t, e), da(e), i & 4 && (vc(
            wa | yu,
            e,
            e.return
          ), pc(wa | yu, e), id(
            e,
            e.return,
            la | yu
          ));
          break;
        case 1:
          Vl(t, e), da(e), i & 512 && (hl || a === null || ka(a, a.return)), i & 64 && Zc && (e = e.updateQueue, e !== null && (i = e.callbacks, i !== null && (a = e.shared.hiddenCallbacks, e.shared.hiddenCallbacks = a === null ? i : a.concat(i))));
          break;
        case 26:
          var o = Wu;
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
                        o = t.createElement(i), kt(o, i, a), t.head.appendChild(o);
                        break;
                      case "meta":
                        if (f = mm(
                          "meta",
                          "content",
                          t
                        ).get(i + (a.content || ""))) {
                          for (d = 0; d < f.length; d++)
                            if (o = f[d], K(
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
          Vl(t, e), da(e), i & 512 && (hl || a === null || ka(a, a.return)), a !== null && i & 4 && xy(
            e,
            e.memoizedProps,
            a.memoizedProps
          );
          break;
        case 5:
          if (Vl(t, e), da(e), i & 512 && (hl || a === null || ka(a, a.return)), e.flags & 32) {
            t = e.stateNode;
            try {
              ye(e, Yu, t);
            } catch (B) {
              Ue(e, e.return, B);
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
          if (Vl(t, e), da(e), i & 4) {
            if (e.stateNode === null)
              throw Error(
                "This should have a text node initialized. This error is likely caused by a bug in React. Please file an issue."
              );
            i = e.memoizedProps, a = a !== null ? a.memoizedProps : i, t = e.stateNode;
            try {
              ye(
                e,
                Uc,
                t,
                a,
                i
              );
            } catch (B) {
              Ue(e, e.return, B);
            }
          }
          break;
        case 3:
          if (o = sn(), sg = null, f = Wu, Wu = br(t.containerInfo), Vl(t, e), Wu = f, da(e), i & 4 && a !== null && a.memoizedState.isDehydrated)
            try {
              ye(
                e,
                rm,
                t.containerInfo
              );
            } catch (B) {
              Ue(e, e.return, B);
            }
          m0 && (m0 = !1, Ec(e)), t.effectDuration += ri(o);
          break;
        case 4:
          i = Wu, Wu = br(
            e.stateNode.containerInfo
          ), Vl(t, e), da(e), Wu = i;
          break;
        case 12:
          i = sn(), Vl(t, e), da(e), e.stateNode.effectDuration += oc(i);
          break;
        case 13:
          Vl(t, e), da(e), e.child.flags & 8192 && e.memoizedState !== null != (a !== null && a.memoizedState !== null) && (T0 = uu()), i & 4 && (i = e.updateQueue, i !== null && (e.updateQueue = null, Tc(e, i)));
          break;
        case 22:
          o = e.memoizedState !== null;
          var h = a !== null && a.memoizedState !== null, v = Zc, b = hl;
          if (Zc = v || o, hl = b || h, Vl(t, e), hl = b, Zc = v, da(e), i & 8192)
            e: for (t = e.stateNode, t._visibility = o ? t._visibility & ~wv : t._visibility | wv, o && (a === null || h || Zc || hl || Xl(e)), a = null, t = e; ; ) {
              if (t.tag === 5 || t.tag === 26) {
                if (a === null) {
                  h = a = t;
                  try {
                    f = h.stateNode, o ? ye(h, ma, f) : ye(
                      h,
                      om,
                      h.stateNode,
                      h.memoizedProps
                    );
                  } catch (B) {
                    Ue(h, h.return, B);
                  }
                }
              } else if (t.tag === 6) {
                if (a === null) {
                  h = t;
                  try {
                    d = h.stateNode, o ? ye(h, cm, d) : ye(
                      h,
                      Ud,
                      d,
                      h.memoizedProps
                    );
                  } catch (B) {
                    Ue(h, h.return, B);
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
          ye(e, uv, e);
        } catch (a) {
          Ue(e, e.return, a);
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
    function Kn(e, t) {
      if (t.subtreeFlags & 8772)
        for (t = t.child; t !== null; )
          wy(e, t.alternate, t), t = t.sibling;
    }
    function Ma(e) {
      switch (e.tag) {
        case 0:
        case 11:
        case 14:
        case 15:
          id(
            e,
            e.return,
            la
          ), Xl(e);
          break;
        case 1:
          ka(e, e.return);
          var t = e.stateNode;
          typeof t.componentWillUnmount == "function" && cd(
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
    function Nu(e, t, a, i) {
      var o = a.flags;
      switch (a.tag) {
        case 0:
        case 11:
        case 15:
          Jn(
            e,
            a,
            i
          ), My(a, la);
          break;
        case 1:
          if (Jn(
            e,
            a,
            i
          ), t = a.stateNode, typeof t.componentDidMount == "function" && ye(
            a,
            o0,
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
              Ue(a, a.return, f);
            }
          }
          i && o & 64 && _y(a), Mo(a, a.return);
          break;
        case 27:
          Ny(a);
        case 26:
        case 5:
          Jn(
            e,
            a,
            i
          ), i && t === null && o & 4 && nv(a), Mo(a, a.return);
          break;
        case 12:
          if (i && o & 4) {
            o = sn(), Jn(
              e,
              a,
              i
            ), i = a.stateNode, i.effectDuration += oc(o);
            try {
              ye(
                a,
                Cy,
                a,
                t,
                jv,
                i.effectDuration
              );
            } catch (f) {
              Ue(a, a.return, f);
            }
          } else
            Jn(
              e,
              a,
              i
            );
          break;
        case 13:
          Jn(
            e,
            a,
            i
          ), i && o & 4 && Uo(e, a);
          break;
        case 22:
          a.memoizedState === null && Jn(
            e,
            a,
            i
          ), Mo(a, a.return);
          break;
        case 30:
          break;
        default:
          Jn(
            e,
            a,
            i
          );
      }
    }
    function Jn(e, t, a) {
      for (a = a && (t.subtreeFlags & 8772) !== 0, t = t.child; t !== null; )
        Nu(
          e,
          t.alternate,
          t,
          a
        ), t = t.sibling;
    }
    function kn(e, t) {
      var a = null;
      e !== null && e.memoizedState !== null && e.memoizedState.cachePool !== null && (a = e.memoizedState.cachePool.pool), e = null, t.memoizedState !== null && t.memoizedState.cachePool !== null && (e = t.memoizedState.cachePool.pool), e !== a && (e != null && cc(e), a != null && Nn(a));
    }
    function bn(e, t) {
      e = null, t.alternate !== null && (e = t.alternate.memoizedState.cache), t = t.memoizedState.cache, t !== e && (cc(t), e != null && Nn(e));
    }
    function Dt(e, t, a, i) {
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
          Dt(
            e,
            t,
            a,
            i
          ), o & 2048 && Uy(t, Yl | yu);
          break;
        case 1:
          Dt(
            e,
            t,
            a,
            i
          );
          break;
        case 3:
          var f = sn();
          Dt(
            e,
            t,
            a,
            i
          ), o & 2048 && (a = null, t.alternate !== null && (a = t.alternate.memoizedState.cache), t = t.memoizedState.cache, t !== a && (cc(t), a != null && Nn(a))), e.passiveEffectDuration += ri(f);
          break;
        case 12:
          if (o & 2048) {
            o = sn(), Dt(
              e,
              t,
              a,
              i
            ), e = t.stateNode, e.passiveEffectDuration += oc(o);
            try {
              ye(
                t,
                av,
                t,
                t.alternate,
                jv,
                e.passiveEffectDuration
              );
            } catch (h) {
              Ue(t, t.return, h);
            }
          } else
            Dt(
              e,
              t,
              a,
              i
            );
          break;
        case 13:
          Dt(
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
          t.memoizedState !== null ? f._visibility & jc ? Dt(
            e,
            t,
            a,
            i
          ) : _o(
            e,
            t
          ) : f._visibility & jc ? Dt(
            e,
            t,
            a,
            i
          ) : (f._visibility |= jc, Ti(
            e,
            t,
            a,
            i,
            (t.subtreeFlags & 10256) !== 0
          )), o & 2048 && kn(d, t);
          break;
        case 24:
          Dt(
            e,
            t,
            a,
            i
          ), o & 2048 && bn(t.alternate, t);
          break;
        default:
          Dt(
            e,
            t,
            a,
            i
          );
      }
    }
    function Ti(e, t, a, i, o) {
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
          Ti(
            e,
            t,
            a,
            i,
            o
          ), Uy(t, Yl);
          break;
        case 23:
          break;
        case 22:
          var d = t.stateNode;
          t.memoizedState !== null ? d._visibility & jc ? Ti(
            e,
            t,
            a,
            i,
            o
          ) : _o(
            e,
            t
          ) : (d._visibility |= jc, Ti(
            e,
            t,
            a,
            i,
            o
          )), o && f & 2048 && kn(
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
    function _o(e, t) {
      if (t.subtreeFlags & 10256)
        for (t = t.child; t !== null; ) {
          var a = e, i = t, o = i.flags;
          switch (i.tag) {
            case 22:
              _o(
                a,
                i
              ), o & 2048 && kn(
                i.alternate,
                i
              );
              break;
            case 24:
              _o(
                a,
                i
              ), o & 2048 && bn(
                i.alternate,
                i
              );
              break;
            default:
              _o(
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
          Ei(e), e = e.sibling;
    }
    function Ei(e) {
      switch (e.tag) {
        case 26:
          Ac(e), e.flags & up && e.memoizedState !== null && pv(
            Wu,
            e.memoizedState,
            e.memoizedProps
          );
          break;
        case 5:
          Ac(e);
          break;
        case 3:
        case 4:
          var t = Wu;
          Wu = br(
            e.stateNode.containerInfo
          ), Ac(e), Wu = t;
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
    function Co(e) {
      var t = e.deletions;
      if ((e.flags & 16) !== 0) {
        if (t !== null)
          for (var a = 0; a < t.length; a++) {
            var i = t[a];
            Jl = i, jy(
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
          Co(e), e.flags & 2048 && or(
            e,
            e.return,
            Yl | yu
          );
          break;
        case 3:
          var t = sn();
          Co(e), e.stateNode.passiveEffectDuration += ri(t);
          break;
        case 12:
          t = sn(), Co(e), e.stateNode.passiveEffectDuration += oc(t);
          break;
        case 22:
          t = e.stateNode, e.memoizedState !== null && t._visibility & jc && (e.return === null || e.return.tag !== 13) ? (t._visibility &= ~jc, dr(e)) : Co(e);
          break;
        default:
          Co(e);
      }
    }
    function dr(e) {
      var t = e.deletions;
      if ((e.flags & 16) !== 0) {
        if (t !== null)
          for (var a = 0; a < t.length; a++) {
            var i = t[a];
            Jl = i, jy(
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
            Yl
          ), dr(e);
          break;
        case 22:
          var t = e.stateNode;
          t._visibility & jc && (t._visibility &= ~jc, dr(e));
          break;
        default:
          dr(e);
      }
    }
    function jy(e, t) {
      for (; Jl !== null; ) {
        var a = Jl, i = a;
        switch (i.tag) {
          case 0:
          case 11:
          case 15:
            or(
              i,
              t,
              Yl
            );
            break;
          case 23:
          case 22:
            i.memoizedState !== null && i.memoizedState.cachePool !== null && (i = i.memoizedState.cachePool.pool, i != null && cc(i));
            break;
          case 24:
            Nn(i.memoizedState.cache);
        }
        if (i = a.child, i !== null) i.return = a, Jl = i;
        else
          e: for (a = e; Jl !== null; ) {
            i = Jl;
            var o = i.sibling, f = i.return;
            if (qy(i), i === a) {
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
    function Gy() {
      JS.forEach(function(e) {
        return e();
      });
    }
    function Ly() {
      var e = typeof IS_REACT_ACT_ENVIRONMENT < "u" ? IS_REACT_ACT_ENVIRONMENT : void 0;
      return e || j.actQueue === null || console.error(
        "The current testing environment is not configured to support act(...)"
      ), e;
    }
    function ha(e) {
      if ((At & qa) !== An && it !== 0)
        return it & -it;
      var t = j.T;
      return t !== null ? (t._updatedFibers || (t._updatedFibers = /* @__PURE__ */ new Set()), t._updatedFibers.add(e), e = Zr, e !== 0 ? e : Wy()) : Ef();
    }
    function iv() {
      On === 0 && (On = (it & 536870912) === 0 || mt ? Je() : 536870912);
      var e = pu.current;
      return e !== null && (e.flags |= 32), On;
    }
    function Kt(e, t, a) {
      if (zh && console.error("useInsertionEffect must not schedule updates."), D0 && (Pv = !0), (e === Ht && (zt === Wr || zt === Fr) || e.cancelPendingCommit !== null) && (Oc(e, 0), wu(
        e,
        it,
        On,
        !1
      )), bu(e, a), (At & qa) !== 0 && e === Ht) {
        if (ba)
          switch (t.tag) {
            case 0:
            case 11:
            case 15:
              e = ut && de(ut) || "Unknown", mb.has(e) || (mb.add(e), t = de(t) || "Unknown", console.error(
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
        Ft && Ya(e, t, a), rv(t), e === Ht && ((At & qa) === An && (sf |= a), ul === $r && wu(
          e,
          it,
          On,
          !1
        )), $a(e);
    }
    function sl(e, t, a) {
      if ((At & (qa | Fu)) !== An)
        throw Error("Should not already be working.");
      var i = !a && (t & 124) === 0 && (t & e.expiredLanes) === 0 || Iu(e, t), o = i ? Xy(e, t) : yd(e, t, !0), f = i;
      do {
        if (o === Kc) {
          Oh && !i && wu(e, t, 0, !1);
          break;
        } else {
          if (a = e.current.alternate, f && !cv(a)) {
            o = yd(e, t, !1), f = !1;
            continue;
          }
          if (o === Ah) {
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
                ).flags |= 256), h = yd(
                  o,
                  h,
                  !1
                ), h !== Ah) {
                  if (b0 && !v) {
                    o.errorRecoveryDisabledLanes |= f, sf |= f, o = $r;
                    break e;
                  }
                  o = Ba, Ba = d, o !== null && (Ba === null ? Ba = o : Ba.push.apply(
                    Ba,
                    o
                  ));
                }
                o = h;
              }
              if (f = !1, o !== Ah) continue;
            }
          }
          if (o === cp) {
            Oc(e, 0), wu(e, t, 0, !0);
            break;
          }
          e: {
            switch (i = e, o) {
              case Kc:
              case cp:
                throw Error("Root did not complete. This is a bug in React.");
              case $r:
                if ((t & 4194048) !== t) break;
              case Wv:
                wu(
                  i,
                  t,
                  On,
                  !ff
                );
                break e;
              case Ah:
                Ba = null;
                break;
              case p0:
              case ib:
                break;
              default:
                throw Error("Unknown root exit status.");
            }
            if (j.actQueue !== null)
              bd(
                i,
                a,
                t,
                Ba,
                dp,
                Fv,
                On,
                sf,
                Ir
              );
            else {
              if ((t & 62914560) === t && (f = T0 + ob - uu(), 10 < f)) {
                if (wu(
                  i,
                  t,
                  On,
                  !ff
                ), pl(i, 0, !0) !== 0) break e;
                i.timeoutHandle = Ab(
                  Sl.bind(
                    null,
                    i,
                    a,
                    Ba,
                    dp,
                    Fv,
                    t,
                    On,
                    sf,
                    Ir,
                    ff,
                    o,
                    FS,
                    r1,
                    0
                  ),
                  f
                );
                break e;
              }
              Sl(
                i,
                a,
                Ba,
                dp,
                Fv,
                t,
                On,
                sf,
                Ir,
                ff,
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
      $a(e);
    }
    function Sl(e, t, a, i, o, f, d, h, v, b, B, L, H, X) {
      if (e.timeoutHandle = as, L = t.subtreeFlags, (L & 8192 || (L & 16785408) === 16785408) && (gp = { stylesheets: null, count: 0, unsuspend: mv }, Ei(t), L = vv(), L !== null)) {
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
            B,
            WS,
            H,
            X
          )
        ), wu(
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
    function wu(e, t, a, i) {
      t &= ~S0, t &= ~sf, e.suspendedLanes |= t, e.pingedLanes &= ~t, i && (e.warmLanes |= t), i = e.expirationTimes;
      for (var o = t; 0 < o; ) {
        var f = 31 - Ql(o), d = 1 << f;
        i[f] = -1, o &= ~d;
      }
      a !== 0 && Tf(e, a, t);
    }
    function Rc() {
      return (At & (qa | Fu)) === An ? (Dc(0), !1) : !0;
    }
    function sd() {
      if (ut !== null) {
        if (zt === an)
          var e = ut.return;
        else
          e = ut, Ms(), mn(e), ph = null, ap = 0, e = ut;
        for (; e !== null; )
          zy(e.alternate, e), e = e.return;
        ut = null;
      }
    }
    function Oc(e, t) {
      var a = e.timeoutHandle;
      a !== as && (e.timeoutHandle = as, rT(a)), a = e.cancelPendingCommit, a !== null && (e.cancelPendingCommit = null, a()), sd(), Ht = e, ut = a = xn(e.current, null), it = t, zt = an, Rn = null, ff = !1, Oh = Iu(e, t), b0 = !1, ul = Kc, Ir = On = S0 = sf = rf = 0, Ba = sp = null, Fv = !1, (t & 8) !== 0 && (t |= t & 32);
      var i = e.entangledLanes;
      if (i !== 0)
        for (e = e.entanglements, i &= t; 0 < i; ) {
          var o = 31 - Ql(i), f = 1 << o;
          t |= e[o], i &= ~f;
        }
      return Vi = t, Nf(), t = o1(), 1e3 < t - c1 && (j.recentlyCreatedOwnerStacks = 0, c1 = t), ku.discardPendingWarnings(), a;
    }
    function yr(e, t) {
      Be = null, j.H = kv, j.getCurrentStack = null, ba = !1, xa = null, t === Pm || t === Xv ? (t = ry(), zt = fp) : t === h1 ? (t = ry(), zt = cb) : zt = t === F1 ? g0 : t !== null && typeof t == "object" && typeof t.then == "function" ? Rh : op, Rn = t;
      var a = ut;
      if (a === null)
        ul = cp, zo(
          e,
          Ra(t, e.current)
        );
      else
        switch (a.mode & ta && Du(a), na(), zt) {
          case op:
            fe !== null && typeof fe.markComponentErrored == "function" && fe.markComponentErrored(
              a,
              t,
              it
            );
            break;
          case Wr:
          case Fr:
          case fp:
          case Rh:
          case rp:
            fe !== null && typeof fe.markComponentSuspended == "function" && fe.markComponentSuspended(
              a,
              t,
              it
            );
        }
    }
    function dd() {
      var e = j.H;
      return j.H = kv, e === null ? kv : e;
    }
    function Vy() {
      var e = j.A;
      return j.A = KS, e;
    }
    function hd() {
      ul = $r, ff || (it & 4194048) !== it && pu.current !== null || (Oh = !0), (rf & 134217727) === 0 && (sf & 134217727) === 0 || Ht === null || wu(
        Ht,
        it,
        On,
        !1
      );
    }
    function yd(e, t, a) {
      var i = At;
      At |= qa;
      var o = dd(), f = Vy();
      if (Ht !== e || it !== t) {
        if (Ft) {
          var d = e.memoizedUpdaters;
          0 < d.size && (wo(e, it), d.clear()), Dl(e, t);
        }
        dp = null, Oc(e, t);
      }
      zn(t), t = !1, d = ul;
      e: do
        try {
          if (zt !== an && ut !== null) {
            var h = ut, v = Rn;
            switch (zt) {
              case g0:
                sd(), d = Wv;
                break e;
              case fp:
              case Wr:
              case Fr:
              case Rh:
                pu.current === null && (t = !0);
                var b = zt;
                if (zt = an, Rn = null, Ai(e, h, v, b), a && Oh) {
                  d = Kc;
                  break e;
                }
                break;
              default:
                b = zt, zt = an, Rn = null, Ai(e, h, v, b);
            }
          }
          md(), d = ul;
          break;
        } catch (B) {
          yr(e, B);
        }
      while (!0);
      return t && e.shellSuspendCounter++, Ms(), At = i, j.H = o, j.A = f, Zi(), ut === null && (Ht = null, it = 0, Nf()), d;
    }
    function md() {
      for (; ut !== null; ) Zy(ut);
    }
    function Xy(e, t) {
      var a = At;
      At |= qa;
      var i = dd(), o = Vy();
      if (Ht !== e || it !== t) {
        if (Ft) {
          var f = e.memoizedUpdaters;
          0 < f.size && (wo(e, it), f.clear()), Dl(e, t);
        }
        dp = null, Iv = uu() + fb, Oc(e, t);
      } else
        Oh = Iu(
          e,
          t
        );
      zn(t);
      e: do
        try {
          if (zt !== an && ut !== null)
            t: switch (t = ut, f = Rn, zt) {
              case op:
                zt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  op
                );
                break;
              case Wr:
              case Fr:
                if (fy(f)) {
                  zt = an, Rn = null, pd(t);
                  break;
                }
                t = function() {
                  zt !== Wr && zt !== Fr || Ht !== e || (zt = rp), $a(e);
                }, f.then(t, t);
                break e;
              case fp:
                zt = rp;
                break e;
              case cb:
                zt = v0;
                break e;
              case rp:
                fy(f) ? (zt = an, Rn = null, pd(t)) : (zt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  rp
                ));
                break;
              case v0:
                var d = null;
                switch (ut.tag) {
                  case 26:
                    d = ut.memoizedState;
                  case 5:
                  case 27:
                    var h = ut;
                    if (!d || Sr(d)) {
                      zt = an, Rn = null;
                      var v = h.sibling;
                      if (v !== null) ut = v;
                      else {
                        var b = h.return;
                        b !== null ? (ut = b, mr(b)) : ut = null;
                      }
                      break t;
                    }
                    break;
                  default:
                    console.error(
                      "Unexpected type of fiber triggered a suspensey commit. This is a bug in React."
                    );
                }
                zt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  v0
                );
                break;
              case Rh:
                zt = an, Rn = null, Ai(
                  e,
                  t,
                  f,
                  Rh
                );
                break;
              case g0:
                sd(), ul = Wv;
                break e;
              default:
                throw Error(
                  "Unexpected SuspendedReason. This is a bug in React."
                );
            }
          j.actQueue !== null ? md() : Qy();
          break;
        } catch (B) {
          yr(e, B);
        }
      while (!0);
      return Ms(), j.H = i, j.A = o, At = a, ut !== null ? (fe !== null && typeof fe.markRenderYielded == "function" && fe.markRenderYielded(), Kc) : (Zi(), Ht = null, it = 0, Nf(), ul);
    }
    function Qy() {
      for (; ut !== null && !Rv(); )
        Zy(ut);
    }
    function Zy(e) {
      var t = e.alternate;
      (e.mode & ta) !== Yt ? (Us(e), t = ye(
        e,
        ud,
        t,
        e,
        Vi
      ), Du(e)) : t = ye(
        e,
        ud,
        t,
        e,
        Vi
      ), e.memoizedProps = e.pendingProps, t === null ? mr(e) : ut = t;
    }
    function pd(e) {
      var t = ye(e, vd, e);
      e.memoizedProps = e.pendingProps, t === null ? mr(e) : ut = t;
    }
    function vd(e) {
      var t = e.alternate, a = (e.mode & ta) !== Yt;
      switch (a && Us(e), e.tag) {
        case 15:
        case 0:
          t = Ey(
            t,
            e,
            e.pendingProps,
            e.type,
            void 0,
            it
          );
          break;
        case 11:
          t = Ey(
            t,
            e,
            e.pendingProps,
            e.type.render,
            e.ref,
            it
          );
          break;
        case 5:
          mn(e);
        default:
          zy(t, e), e = ut = Wh(e, Vi), t = ud(t, e, Vi);
      }
      return a && Du(e), t;
    }
    function Ai(e, t, a, i) {
      Ms(), mn(t), ph = null, ap = 0;
      var o = t.return;
      try {
        if (tr(
          e,
          o,
          t,
          a,
          it
        )) {
          ul = cp, zo(
            e,
            Ra(a, e.current)
          ), ut = null;
          return;
        }
      } catch (f) {
        if (o !== null) throw ut = o, f;
        ul = cp, zo(
          e,
          Ra(a, e.current)
        ), ut = null;
        return;
      }
      t.flags & 32768 ? (mt || i === op ? e = !0 : Oh || (it & 536870912) !== 0 ? e = !1 : (ff = e = !0, (i === Wr || i === Fr || i === fp || i === Rh) && (i = pu.current, i !== null && i.tag === 13 && (i.flags |= 16384))), gd(t, e)) : mr(t);
    }
    function mr(e) {
      var t = e;
      do {
        if ((t.flags & 32768) !== 0) {
          gd(
            t,
            ff
          );
          return;
        }
        var a = t.alternate;
        if (e = t.return, Us(t), a = ye(
          t,
          Pp,
          a,
          t,
          Vi
        ), (t.mode & ta) !== Yt && fc(t), a !== null) {
          ut = a;
          return;
        }
        if (t = t.sibling, t !== null) {
          ut = t;
          return;
        }
        ut = t = e;
      } while (t !== null);
      ul === Kc && (ul = ib);
    }
    function gd(e, t) {
      do {
        var a = ev(e.alternate, e);
        if (a !== null) {
          a.flags &= 32767, ut = a;
          return;
        }
        if ((e.mode & ta) !== Yt) {
          fc(e), a = e.actualDuration;
          for (var i = e.child; i !== null; )
            a += i.actualDuration, i = i.sibling;
          e.actualDuration = a;
        }
        if (a = e.return, a !== null && (a.flags |= 32768, a.subtreeFlags = 0, a.deletions = null), !t && (e = e.sibling, e !== null)) {
          ut = e;
          return;
        }
        ut = e = a;
      } while (e !== null);
      ul = Wv, ut = null;
    }
    function bd(e, t, a, i, o, f, d, h, v) {
      e.cancelPendingCommit = null;
      do
        xo();
      while (aa !== Pr);
      if (ku.flushLegacyContextWarning(), ku.flushPendingUnsafeLifecycleWarnings(), (At & (qa | Fu)) !== An)
        throw Error("Should not already be working.");
      if (fe !== null && typeof fe.markCommitStarted == "function" && fe.markCommitStarted(a), t === null) Ne();
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
        ), e === Ht && (ut = Ht = null, it = 0), Dh = t, hf = e, yf = a, A0 = f, R0 = o, hb = i, (t.subtreeFlags & 10256) !== 0 || (t.flags & 10256) !== 0 ? (e.callbackNode = null, e.callbackPriority = 0, $y(Wo, function() {
          return pr(), null;
        })) : (e.callbackNode = null, e.callbackPriority = 0), jv = sh(), i = (t.flags & 13878) !== 0, (t.subtreeFlags & 13878) !== 0 || i) {
          i = j.T, j.T = null, o = xe.p, xe.p = ql, d = At, At |= Fu;
          try {
            od(e, t, a);
          } finally {
            At = d, xe.p = o, j.T = i;
          }
        }
        aa = rb, $n(), Sd(), ov();
      }
    }
    function $n() {
      if (aa === rb) {
        aa = Pr;
        var e = hf, t = Dh, a = yf, i = (t.flags & 13878) !== 0;
        if ((t.subtreeFlags & 13878) !== 0 || i) {
          i = j.T, j.T = null;
          var o = xe.p;
          xe.p = ql;
          var f = At;
          At |= Fu;
          try {
            Th = a, Eh = e, By(t, e), Eh = Th = null, a = w0;
            var d = jp(e.containerInfo), h = a.focusedElem, v = a.selectionRange;
            if (d !== h && h && h.ownerDocument && Yp(
              h.ownerDocument.documentElement,
              h
            )) {
              if (v !== null && Kh(h)) {
                var b = v.start, B = v.end;
                if (B === void 0 && (B = b), "selectionStart" in h)
                  h.selectionStart = b, h.selectionEnd = Math.min(
                    B,
                    h.value.length
                  );
                else {
                  var L = h.ownerDocument || document, H = L && L.defaultView || window;
                  if (H.getSelection) {
                    var X = H.getSelection(), me = h.textContent.length, He = Math.min(
                      v.start,
                      me
                    ), Nt = v.end === void 0 ? He : Math.min(v.end, me);
                    !X.extend && He > Nt && (d = Nt, Nt = He, He = d);
                    var ft = Zh(
                      h,
                      He
                    ), T = Zh(
                      h,
                      Nt
                    );
                    if (ft && T && (X.rangeCount !== 1 || X.anchorNode !== ft.node || X.anchorOffset !== ft.offset || X.focusNode !== T.node || X.focusOffset !== T.offset)) {
                      var E = L.createRange();
                      E.setStart(ft.node, ft.offset), X.removeAllRanges(), He > Nt ? (X.addRange(E), X.extend(T.node, T.offset)) : (E.setEnd(T.node, T.offset), X.addRange(E));
                    }
                  }
                }
              }
              for (L = [], X = h; X = X.parentNode; )
                X.nodeType === 1 && L.push({
                  element: X,
                  left: X.scrollLeft,
                  top: X.scrollTop
                });
              for (typeof h.focus == "function" && h.focus(), h = 0; h < L.length; h++) {
                var A = L[h];
                A.element.scrollLeft = A.left, A.element.scrollTop = A.top;
              }
            }
            yg = !!N0, w0 = N0 = null;
          } finally {
            At = f, xe.p = o, j.T = i;
          }
        }
        e.current = t, aa = sb;
      }
    }
    function Sd() {
      if (aa === sb) {
        aa = Pr;
        var e = hf, t = Dh, a = yf, i = (t.flags & 8772) !== 0;
        if ((t.subtreeFlags & 8772) !== 0 || i) {
          i = j.T, j.T = null;
          var o = xe.p;
          xe.p = ql;
          var f = At;
          At |= Fu;
          try {
            fe !== null && typeof fe.markLayoutEffectsStarted == "function" && fe.markLayoutEffectsStarted(a), Th = a, Eh = e, wy(
              e,
              t.alternate,
              t
            ), Eh = Th = null, fe !== null && typeof fe.markLayoutEffectsStopped == "function" && fe.markLayoutEffectsStopped();
          } finally {
            At = f, xe.p = o, j.T = i;
          }
        }
        aa = db;
      }
    }
    function ov() {
      if (aa === IS || aa === db) {
        aa = Pr, Yg();
        var e = hf, t = Dh, a = yf, i = hb, o = (t.subtreeFlags & 10256) !== 0 || (t.flags & 10256) !== 0;
        o ? aa = E0 : (aa = Pr, Dh = hf = null, Wn(e, e.pendingLanes), es = 0, yp = null);
        var f = e.pendingLanes;
        if (f === 0 && (df = null), o || No(e), o = to(a), t = t.stateNode, wl && typeof wl.onCommitFiberRoot == "function")
          try {
            var d = (t.current.flags & 128) === 128;
            switch (o) {
              case ql:
                var h = Xd;
                break;
              case En:
                h = xr;
                break;
              case Qu:
                h = Wo;
                break;
              case Jd:
                h = Hr;
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
          } catch (L) {
            va || (va = !0, console.error(
              "React instrumentation encountered an error: %s",
              L
            ));
          }
        if (Ft && e.memoizedUpdaters.clear(), Gy(), i !== null) {
          d = j.T, h = xe.p, xe.p = ql, j.T = null;
          try {
            var v = e.onRecoverableError;
            for (t = 0; t < i.length; t++) {
              var b = i[t], B = fv(b.stack);
              ye(
                b.source,
                v,
                b.value,
                B
              );
            }
          } finally {
            j.T = d, xe.p = h;
          }
        }
        (yf & 3) !== 0 && xo(), $a(e), f = e.pendingLanes, (a & 4194090) !== 0 && (f & 42) !== 0 ? (Lv = !0, e === O0 ? hp++ : (hp = 0, O0 = e)) : hp = 0, Dc(0), Ne();
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
    function Wn(e, t) {
      (e.pooledCacheLanes &= t) === 0 && (t = e.pooledCache, t != null && (e.pooledCache = null, Nn(t)));
    }
    function xo(e) {
      return $n(), Sd(), ov(), pr();
    }
    function pr() {
      if (aa !== E0) return !1;
      var e = hf, t = A0;
      A0 = 0;
      var a = to(yf), i = Qu > a ? Qu : a;
      a = j.T;
      var o = xe.p;
      try {
        xe.p = i, j.T = null, i = R0, R0 = null;
        var f = hf, d = yf;
        if (aa = Pr, Dh = hf = null, yf = 0, (At & (qa | Fu)) !== An)
          throw Error("Cannot flush passive effects while already rendering.");
        D0 = !0, Pv = !1, fe !== null && typeof fe.markPassiveEffectsStarted == "function" && fe.markPassiveEffectsStarted(d);
        var h = At;
        if (At |= Fu, Yy(f.current), rr(
          f,
          f.current,
          d,
          i
        ), fe !== null && typeof fe.markPassiveEffectsStopped == "function" && fe.markPassiveEffectsStopped(), No(f), At = h, Dc(0, !1), Pv ? f === yp ? es++ : (es = 0, yp = f) : es = 0, Pv = D0 = !1, wl && typeof wl.onPostCommitFiberRoot == "function")
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
        xe.p = o, j.T = a, Wn(e, t);
      }
    }
    function Ho(e, t, a) {
      t = Ra(a, t), t = Ll(e.stateNode, t, 2), e = hn(e, t, 2), e !== null && (bu(e, 2), $a(e));
    }
    function Ue(e, t, a) {
      if (zh = !1, e.tag === 3)
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
              e = Ra(a, e), a = Zt(2), i = hn(t, a, 2), i !== null && (er(
                a,
                i,
                t,
                e
              ), bu(i, 2), $a(i));
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
      o.has(a) || (b0 = !0, o.add(a), i = Cg.bind(null, e, t, a), Ft && wo(e, a), t.then(i, i));
    }
    function Cg(e, t, a) {
      var i = e.pingCache;
      i !== null && i.delete(t), e.pingedLanes |= e.suspendedLanes & a, e.warmLanes &= ~a, Ly() && j.actQueue === null && console.error(
        `A suspended resource finished loading inside a test, but the event was not wrapped in act(...).

When testing, code that resolves suspended data should be wrapped into act(...):

act(() => {
  /* finish loading suspended data */
});
/* assert on the output */

This ensures that you're testing the behavior the user would see in the browser. Learn more at https://react.dev/link/wrap-tests-with-act`
      ), Ht === e && (it & a) === a && (ul === $r || ul === p0 && (it & 62914560) === it && uu() - T0 < ob ? (At & qa) === An && Oc(e, 0) : S0 |= a, Ir === it && (Ir = 0)), $a(e);
    }
    function Jy(e, t) {
      t === 0 && (t = Un()), e = ia(e, t), e !== null && (bu(e, t), $a(e));
    }
    function vr(e) {
      var t = e.memoizedState, a = 0;
      t !== null && (a = t.retryLane), Jy(e, a);
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
      i !== null && i.delete(t), Jy(e, a);
    }
    function Td(e, t, a) {
      if ((t.subtreeFlags & 67117056) !== 0)
        for (t = t.child; t !== null; ) {
          var i = e, o = t, f = o.type === Zo;
          f = a || f, o.tag !== 22 ? o.flags & 67108864 ? f && ye(
            o,
            ky,
            i,
            o,
            (o.mode & n1) === Yt
          ) : Td(
            i,
            o,
            f
          ) : o.memoizedState === null && (f && o.flags & 8192 ? ye(
            o,
            ky,
            i,
            o
          ) : o.subtreeFlags & 67108864 && ye(
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
        Ma(t), a && hr(t), Nu(e, t.alternate, t, !1), a && rd(e, t, 0, null, !1, 0);
      } finally {
        oe(!1);
      }
    }
    function No(e) {
      var t = !0;
      e.current.mode & (Sa | Ju) || (t = !1), Td(
        e,
        e.current,
        t
      );
    }
    function Sn(e) {
      if ((At & qa) === An) {
        var t = e.tag;
        if (t === 3 || t === 1 || t === 0 || t === 11 || t === 14 || t === 15) {
          if (t = de(e) || "ReactComponent", eg !== null) {
            if (eg.has(t)) return;
            eg.add(t);
          } else eg = /* @__PURE__ */ new Set([t]);
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
        Ya(e, a, t);
      });
    }
    function $y(e, t) {
      var a = j.actQueue;
      return a !== null ? (a.push(t), tT) : Vd(e, t);
    }
    function rv(e) {
      Ly() && j.actQueue === null && ye(e, function() {
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
    function $a(e) {
      e !== Mh && e.next === null && (Mh === null ? tg = Mh = e : Mh = Mh.next = e), lg = !0, j.actQueue !== null ? M0 || (M0 = !0, dl()) : z0 || (z0 = !0, dl());
    }
    function Dc(e, t) {
      if (!U0 && lg) {
        U0 = !0;
        do
          for (var a = !1, i = tg; i !== null; ) {
            if (e !== 0) {
              var o = i.pendingLanes;
              if (o === 0) var f = 0;
              else {
                var d = i.suspendedLanes, h = i.pingedLanes;
                f = (1 << 31 - Ql(42 | e) + 1) - 1, f &= o & ~(d & ~h), f = f & 201326741 ? f & 201326741 | 1 : f ? f | 2 : 0;
              }
              f !== 0 && (a = !0, Rd(i, f));
            } else
              f = it, f = pl(
                i,
                i === Ht ? f : 0,
                i.cancelPendingCommit !== null || i.timeoutHandle !== as
              ), (f & 3) === 0 || Iu(i, f) || (a = !0, Rd(i, f));
            i = i.next;
          }
        while (a);
        U0 = !1;
      }
    }
    function Ed() {
      Ad();
    }
    function Ad() {
      lg = M0 = z0 = !1;
      var e = 0;
      ts !== 0 && (qo() && (e = ts), ts = 0);
      for (var t = uu(), a = null, i = tg; i !== null; ) {
        var o = i.next, f = Fn(i, t);
        f === 0 ? (i.next = null, a === null ? tg = o : a.next = o, o === null && (Mh = a)) : (a = i, (e !== 0 || (f & 3) !== 0) && (lg = !0)), i = o;
      }
      Dc(e);
    }
    function Fn(e, t) {
      for (var a = e.suspendedLanes, i = e.pingedLanes, o = e.expirationTimes, f = e.pendingLanes & -62914561; 0 < f; ) {
        var d = 31 - Ql(f), h = 1 << d, v = o[d];
        v === -1 ? ((h & a) === 0 || (h & i) !== 0) && (o[d] = cs(h, t)) : v <= t && (e.expiredLanes |= h), f &= ~h;
      }
      if (t = Ht, a = it, a = pl(
        e,
        e === t ? a : 0,
        e.cancelPendingCommit !== null || e.timeoutHandle !== as
      ), i = e.callbackNode, a === 0 || e === t && (zt === Wr || zt === Fr) || e.cancelPendingCommit !== null)
        return i !== null && Od(i), e.callbackNode = null, e.callbackPriority = 0;
      if ((a & 3) === 0 || Iu(e, a)) {
        if (t = a & -a, t !== e.callbackPriority || j.actQueue !== null && i !== _0)
          Od(i);
        else return t;
        switch (to(a)) {
          case ql:
          case En:
            a = xr;
            break;
          case Qu:
            a = Wo;
            break;
          case Jd:
            a = Hr;
            break;
          default:
            a = Wo;
        }
        return i = Jt.bind(null, e), j.actQueue !== null ? (j.actQueue.push(i), a = _0) : a = Vd(a, i), e.callbackPriority = t, e.callbackNode = a, t;
      }
      return i !== null && Od(i), e.callbackPriority = 2, e.callbackNode = null, 2;
    }
    function Jt(e, t) {
      if (Lv = Gv = !1, aa !== Pr && aa !== E0)
        return e.callbackNode = null, e.callbackPriority = 0, null;
      var a = e.callbackNode;
      if (xo() && e.callbackNode !== a)
        return null;
      var i = it;
      return i = pl(
        e,
        e === Ht ? i : 0,
        e.cancelPendingCommit !== null || e.timeoutHandle !== as
      ), i === 0 ? null : (sl(
        e,
        i,
        t
      ), Fn(e, uu()), e.callbackNode != null && e.callbackNode === a ? Jt.bind(null, e) : null);
    }
    function Rd(e, t) {
      if (xo()) return null;
      Gv = Lv, Lv = !1, sl(e, t, !0);
    }
    function Od(e) {
      e !== _0 && e !== null && Bg(e);
    }
    function dl() {
      j.actQueue !== null && j.actQueue.push(function() {
        return Ad(), null;
      }), sT(function() {
        (At & (qa | Fu)) !== An ? Vd(
          Xd,
          Ed
        ) : Ad();
      });
    }
    function Wy() {
      return ts === 0 && (ts = Je()), ts;
    }
    function Fy(e) {
      return e == null || typeof e == "symbol" || typeof e == "boolean" ? null : typeof e == "function" ? e : (K(e, "action"), oo("" + e));
    }
    function Iy(e, t) {
      var a = t.ownerDocument.createElement("input");
      return a.name = t.name, a.value = t.value, e.id && a.setAttribute("form", e.id), t.parentNode.insertBefore(a, t), e = new FormData(e), a.parentNode.removeChild(a), e;
    }
    function qt(e, t, a, i, o) {
      if (t === "submit" && a && a.stateNode === o) {
        var f = Fy(
          (o[ga] || null).action
        ), d = i.submitter;
        d && (t = (t = d[ga] || null) ? Fy(t.formAction) : d.getAttribute("formAction"), t !== null && (f = t, d = null));
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
                    Object.freeze(b), hc(
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
    function In(e, t) {
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
        case ql:
          var o = Ng;
          break;
        case En:
          o = Bd;
          break;
        default:
          o = Mi;
      }
      a = o.bind(
        null,
        t,
        a,
        e
      ), o = void 0, !x || t !== "touchstart" && t !== "touchmove" && t !== "wheel" || (o = !0), i ? o !== void 0 ? e.addEventListener(t, a, {
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
      gs(function() {
        var b = f, B = Pi(a), L = [];
        e: {
          var H = a1.get(e);
          if (H !== void 0) {
            var X = Re, me = e;
            switch (e) {
              case "keypress":
                if (fo(a) === 0) break e;
              case "keydown":
              case "keyup":
                X = gS;
                break;
              case "focusin":
                me = "focus", X = rt;
                break;
              case "focusout":
                me = "blur", X = rt;
                break;
              case "beforeblur":
              case "afterblur":
                X = rt;
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
                X = Fe;
                break;
              case "drag":
              case "dragend":
              case "dragenter":
              case "dragexit":
              case "dragleave":
              case "dragover":
              case "dragstart":
              case "drop":
                X = _e;
                break;
              case "touchcancel":
              case "touchend":
              case "touchmove":
              case "touchstart":
                X = TS;
                break;
              case P0:
              case e1:
              case t1:
                X = Vg;
                break;
              case l1:
                X = AS;
                break;
              case "scroll":
              case "scrollend":
                X = z;
                break;
              case "wheel":
                X = OS;
                break;
              case "copy":
              case "cut":
              case "paste":
                X = sS;
                break;
              case "gotpointercapture":
              case "lostpointercapture":
              case "pointercancel":
              case "pointerdown":
              case "pointermove":
              case "pointerout":
              case "pointerover":
              case "pointerup":
                X = Z0;
                break;
              case "toggle":
              case "beforetoggle":
                X = zS;
            }
            var He = (t & 4) !== 0, Nt = !He && (e === "scroll" || e === "scrollend"), ft = He ? H !== null ? H + "Capture" : null : H;
            He = [];
            for (var T = b, E; T !== null; ) {
              var A = T;
              if (E = A.stateNode, A = A.tag, A !== 5 && A !== 26 && A !== 27 || E === null || ft === null || (A = Eu(T, ft), A != null && He.push(
                Il(
                  T,
                  A,
                  E
                )
              )), Nt) break;
              T = T.return;
            }
            0 < He.length && (H = new X(
              H,
              me,
              null,
              a,
              B
            ), L.push({
              event: H,
              listeners: He
            }));
          }
        }
        if ((t & 7) === 0) {
          e: {
            if (H = e === "mouseover" || e === "pointerover", X = e === "mouseout" || e === "pointerout", H && a !== r && (me = a.relatedTarget || a.fromElement) && (ua(me) || me[wi]))
              break e;
            if ((X || H) && (H = B.window === B ? B : (H = B.ownerDocument) ? H.defaultView || H.parentWindow : window, X ? (me = a.relatedTarget || a.toElement, X = b, me = me ? ua(me) : null, me !== null && (Nt = tt(me), He = me.tag, me !== Nt || He !== 5 && He !== 27 && He !== 6) && (me = null)) : (X = null, me = b), X !== me)) {
              if (He = Fe, A = "onMouseLeave", ft = "onMouseEnter", T = "mouse", (e === "pointerout" || e === "pointerover") && (He = Z0, A = "onPointerLeave", ft = "onPointerEnter", T = "pointer"), Nt = X == null ? H : un(X), E = me == null ? H : un(me), H = new He(
                A,
                T + "leave",
                X,
                a,
                B
              ), H.target = Nt, H.relatedTarget = E, A = null, ua(B) === b && (He = new He(
                ft,
                T + "enter",
                me,
                a,
                B
              ), He.target = E, He.relatedTarget = Nt, A = He), Nt = A, X && me)
                t: {
                  for (He = X, ft = me, T = 0, E = He; E; E = Tl(E))
                    T++;
                  for (E = 0, A = ft; A; A = Tl(A))
                    E++;
                  for (; 0 < T - E; )
                    He = Tl(He), T--;
                  for (; 0 < E - T; )
                    ft = Tl(ft), E--;
                  for (; T--; ) {
                    if (He === ft || ft !== null && He === ft.alternate)
                      break t;
                    He = Tl(He), ft = Tl(ft);
                  }
                  He = null;
                }
              else He = null;
              X !== null && em(
                L,
                H,
                X,
                He,
                !1
              ), me !== null && Nt !== null && em(
                L,
                Nt,
                me,
                He,
                !0
              );
            }
          }
          e: {
            if (H = b ? un(b) : window, X = H.nodeName && H.nodeName.toLowerCase(), X === "select" || X === "input" && H.type === "file")
              var Q = Xh;
            else if (Np(H))
              if (F0)
                Q = zg;
              else {
                Q = Qh;
                var ne = Og;
              }
            else
              X = H.nodeName, !X || X.toLowerCase() !== "input" || H.type !== "checkbox" && H.type !== "radio" ? b && Ii(b.elementType) && (Q = Xh) : Q = Dg;
            if (Q && (Q = Q(e, b))) {
              Es(
                L,
                Q,
                a,
                B
              );
              break e;
            }
            ne && ne(e, H, b), e === "focusout" && b && H.type === "number" && b.memoizedProps.value != null && ds(H, "number", H.value);
          }
          switch (ne = b ? un(b) : window, e) {
            case "focusin":
              (Np(ne) || ne.contentEditable === "true") && (ah = ne, Qg = b, Zm = null);
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
                B
              );
              break;
            case "selectionchange":
              if (CS) break;
            case "keydown":
            case "keyup":
              Gp(
                L,
                a,
                B
              );
          }
          var Qe;
          if (Xg)
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
            lh ? Wl(e, a) && (pe = "onCompositionEnd") : e === "keydown" && a.keyCode === K0 && (pe = "onCompositionStart");
          pe && (J0 && a.locale !== "ko" && (lh || pe !== "onCompositionStart" ? pe === "onCompositionEnd" && lh && (Qe = Au()) : ($ = B, w = "value" in $ ? $.value : $.textContent, lh = !0)), ne = gr(
            b,
            pe
          ), 0 < ne.length && (pe = new Q0(
            pe,
            e,
            null,
            a,
            B
          ), L.push({
            event: pe,
            listeners: ne
          }), Qe ? pe.data = Qe : (Qe = ni(a), Qe !== null && (pe.data = Qe)))), (Qe = US ? Ts(e, a) : Cf(e, a)) && (pe = gr(
            b,
            "onBeforeInput"
          ), 0 < pe.length && (ne = new hS(
            "onBeforeInput",
            "beforeinput",
            null,
            a,
            B
          ), L.push({
            event: ne,
            listeners: pe
          }), ne.data = Qe)), qt(
            L,
            e,
            b,
            a,
            B
          );
        }
        In(L, t);
      });
    }
    function Il(e, t, a) {
      return {
        instance: e,
        listener: t,
        currentTarget: a
      };
    }
    function gr(e, t) {
      for (var a = t + "Capture", i = []; e !== null; ) {
        var o = e, f = o.stateNode;
        if (o = o.tag, o !== 5 && o !== 26 && o !== 27 || f === null || (o = Eu(e, a), o != null && i.unshift(
          Il(e, o, f)
        ), o = Eu(e, t), o != null && i.push(
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
    function em(e, t, a, i, o) {
      for (var f = t._reactName, d = []; a !== null && a !== i; ) {
        var h = a, v = h.alternate, b = h.stateNode;
        if (h = h.tag, v !== null && v === i) break;
        h !== 5 && h !== 26 && h !== 27 || b === null || (v = b, o ? (b = Eu(a, f), b != null && d.unshift(
          Il(a, b, v)
        )) : o || (b = Eu(a, f), b != null && d.push(
          Il(a, b, v)
        ))), a = a.return;
      }
      d.length !== 0 && e.push({ event: t, listeners: d });
    }
    function Pn(e, t) {
      co(e, t), e !== "input" && e !== "textarea" && e !== "select" || t == null || t.value !== null || Lm || (Lm = !0, e === "select" && t.multiple ? console.error(
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
      Ii(e) || typeof t.is == "string" || Gh(e, t, a), t.contentEditable && !t.suppressContentEditableWarning && t.children != null && console.error(
        "A component is `contentEditable` and contains `children` managed by React. It is now your responsibility to guarantee that none of those nodes are unexpectedly modified or duplicated. This is probably not intentional."
      );
    }
    function Bt(e, t, a, i) {
      t !== a && (a = xl(a), xl(t) !== a && (i[e] = t));
    }
    function Oi(e, t, a) {
      t.forEach(function(i) {
        a[lm(i)] = i === "style" ? Mc(e) : e.getAttribute(i);
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
    function Md(e, t) {
      return e = e.namespaceURI === Gr || e.namespaceURI === af ? e.ownerDocument.createElementNS(
        e.namespaceURI,
        e.tagName
      ) : e.ownerDocument.createElement(e.tagName), e.innerHTML = t, e.innerHTML;
    }
    function xl(e) {
      return g(e) && (console.error(
        "The provided HTML markup uses a value of unsupported type %s. This value must be coerced to a string before using it here.",
        be(e)
      ), q(e)), (typeof e == "string" ? e : "" + e).replace(lT, `
`).replace(aT, "");
    }
    function tm(e, t) {
      return t = xl(t), xl(e) === t;
    }
    function qu() {
    }
    function ht(e, t, a, i, o, f) {
      switch (a) {
        case "children":
          typeof i == "string" ? (Uf(i, t, !1), t === "body" || t === "textarea" && i === "" || Fi(e, i)) : (typeof i == "number" || typeof i == "bigint") && (Uf("" + i, t, !1), t !== "body" && Fi(e, "" + i));
          break;
        case "className":
          je(e, "class", i);
          break;
        case "tabIndex":
          je(e, "tabindex", i);
          break;
        case "dir":
        case "role":
        case "viewBox":
        case "width":
        case "height":
          je(e, a, i);
          break;
        case "style":
          _f(e, i, f);
          break;
        case "data":
          if (t !== "object") {
            je(e, "data", i);
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
          K(i, a), i = oo("" + i), e.setAttribute(a, i);
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
          K(i, a), i = oo("" + i), e.setAttribute(a, i);
          break;
        case "onClick":
          i != null && (typeof i != "function" && Wa(a, i), e.onclick = qu);
          break;
        case "onScroll":
          i != null && (typeof i != "function" && Wa(a, i), et("scroll", e));
          break;
        case "onScrollEnd":
          i != null && (typeof i != "function" && Wa(a, i), et("scrollend", e));
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
          K(i, a), a = oo("" + i), e.setAttributeNS(ls, "xlink:href", a);
          break;
        case "contentEditable":
        case "spellCheck":
        case "draggable":
        case "value":
        case "autoReverse":
        case "externalResourcesRequired":
        case "focusable":
        case "preserveAlpha":
          i != null && typeof i != "function" && typeof i != "symbol" ? (K(i, a), e.setAttribute(a, "" + i)) : e.removeAttribute(a);
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
          i === !0 ? e.setAttribute(a, "") : i !== !1 && i != null && typeof i != "function" && typeof i != "symbol" ? (K(i, a), e.setAttribute(a, i)) : e.removeAttribute(a);
          break;
        case "cols":
        case "rows":
        case "size":
        case "span":
          i != null && typeof i != "function" && typeof i != "symbol" && !isNaN(i) && 1 <= i ? (K(i, a), e.setAttribute(a, i)) : e.removeAttribute(a);
          break;
        case "rowSpan":
        case "start":
          i == null || typeof i == "function" || typeof i == "symbol" || isNaN(i) ? e.removeAttribute(a) : (K(i, a), e.setAttribute(a, i));
          break;
        case "popover":
          et("beforetoggle", e), et("toggle", e), ct(e, "popover", i);
          break;
        case "xlinkActuate":
          ll(
            e,
            ls,
            "xlink:actuate",
            i
          );
          break;
        case "xlinkArcrole":
          ll(
            e,
            ls,
            "xlink:arcrole",
            i
          );
          break;
        case "xlinkRole":
          ll(
            e,
            ls,
            "xlink:role",
            i
          );
          break;
        case "xlinkShow":
          ll(
            e,
            ls,
            "xlink:show",
            i
          );
          break;
        case "xlinkTitle":
          ll(
            e,
            ls,
            "xlink:title",
            i
          );
          break;
        case "xlinkType":
          ll(
            e,
            ls,
            "xlink:type",
            i
          );
          break;
        case "xmlBase":
          ll(
            e,
            x0,
            "xml:base",
            i
          );
          break;
        case "xmlLang":
          ll(
            e,
            x0,
            "xml:lang",
            i
          );
          break;
        case "xmlSpace":
          ll(
            e,
            x0,
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
          bb || i == null || typeof i != "object" || (bb = !0, console.error(
            "The `popoverTarget` prop expects the ID of an Element as a string. Received %s instead.",
            i
          ));
        default:
          !(2 < a.length) || a[0] !== "o" && a[0] !== "O" || a[1] !== "n" && a[1] !== "N" ? (a = vs(a), ct(e, a, i)) : en.hasOwnProperty(a) && i != null && typeof i != "function" && Wa(a, i);
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
          i != null && (typeof i != "function" && Wa(a, i), et("scroll", e));
          break;
        case "onScrollEnd":
          i != null && (typeof i != "function" && Wa(a, i), et("scrollend", e));
          break;
        case "onClick":
          i != null && (typeof i != "function" && Wa(a, i), e.onclick = qu);
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
      switch (Pn(t, a), t) {
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
          ve("input", a), et("invalid", e);
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
          ei(e, a), Up(
            e,
            f,
            h,
            v,
            b,
            d,
            o,
            !1
          ), Su(e);
          return;
        case "select":
          ve("select", a), et("invalid", e), i = d = f = null;
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
          Df(e, a), t = f, a = d, e.multiple = !!i, t != null ? Tu(e, !!i, t, !1) : a != null && Tu(e, !!i, a, !0);
          return;
        case "textarea":
          ve("textarea", a), et("invalid", e), f = o = i = null;
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
          _n(e, a), wh(e, i, o, f), Su(e);
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
      switch (Pn(t, i), t) {
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
          for (X in a) {
            var L = a[X];
            if (a.hasOwnProperty(X) && L != null)
              switch (X) {
                case "checked":
                  break;
                case "value":
                  break;
                case "defaultValue":
                  v = L;
                default:
                  i.hasOwnProperty(X) || ht(
                    e,
                    t,
                    X,
                    null,
                    i,
                    L
                  );
              }
          }
          for (var H in i) {
            var X = i[H];
            if (L = a[H], i.hasOwnProperty(H) && (X != null || L != null))
              switch (H) {
                case "type":
                  f = X;
                  break;
                case "name":
                  o = X;
                  break;
                case "checked":
                  b = X;
                  break;
                case "defaultChecked":
                  B = X;
                  break;
                case "value":
                  d = X;
                  break;
                case "defaultValue":
                  h = X;
                  break;
                case "children":
                case "dangerouslySetInnerHTML":
                  if (X != null)
                    throw Error(
                      t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                    );
                  break;
                default:
                  X !== L && ht(
                    e,
                    t,
                    H,
                    X,
                    i,
                    L
                  );
              }
          }
          t = a.type === "checkbox" || a.type === "radio" ? a.checked != null : a.value != null, i = i.type === "checkbox" || i.type === "radio" ? i.checked != null : i.value != null, t || !i || vb || (console.error(
            "A component is changing an uncontrolled input to be controlled. This is likely caused by the value changing from undefined to a defined value, which should not happen. Decide between using a controlled or uncontrolled input element for the lifetime of the component. More info: https://react.dev/link/controlled-components"
          ), vb = !0), !t || i || pb || (console.error(
            "A component is changing a controlled input to be uncontrolled. This is likely caused by the value changing from a defined to undefined, which should not happen. Decide between using a controlled or uncontrolled input element for the lifetime of the component. More info: https://react.dev/link/controlled-components"
          ), pb = !0), ti(
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
          X = d = h = H = null;
          for (f in a)
            if (v = a[f], a.hasOwnProperty(f) && v != null)
              switch (f) {
                case "value":
                  break;
                case "multiple":
                  X = v;
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
                  H = f;
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
          i = h, t = d, a = X, H != null ? Tu(e, !!t, H, !1) : !!a != !!t && (i != null ? Tu(e, !!t, i, !0) : Tu(e, !!t, t ? [] : "", !1));
          return;
        case "textarea":
          X = H = null;
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
                  H = o;
                  break;
                case "defaultValue":
                  X = o;
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
          hs(e, H, X);
          return;
        case "option":
          for (var me in a)
            if (H = a[me], a.hasOwnProperty(me) && H != null && !i.hasOwnProperty(me))
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
                    H
                  );
              }
          for (v in i)
            if (H = i[v], X = a[v], i.hasOwnProperty(v) && H !== X && (H != null || X != null))
              switch (v) {
                case "selected":
                  e.selected = H && typeof H != "function" && typeof H != "symbol";
                  break;
                default:
                  ht(
                    e,
                    t,
                    v,
                    H,
                    i,
                    X
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
          for (var He in a)
            H = a[He], a.hasOwnProperty(He) && H != null && !i.hasOwnProperty(He) && ht(
              e,
              t,
              He,
              null,
              i,
              H
            );
          for (b in i)
            if (H = i[b], X = a[b], i.hasOwnProperty(b) && H !== X && (H != null || X != null))
              switch (b) {
                case "children":
                case "dangerouslySetInnerHTML":
                  if (H != null)
                    throw Error(
                      t + " is a void element tag and must neither have `children` nor use `dangerouslySetInnerHTML`."
                    );
                  break;
                default:
                  ht(
                    e,
                    t,
                    b,
                    H,
                    i,
                    X
                  );
              }
          return;
        default:
          if (Ii(t)) {
            for (var Nt in a)
              H = a[Nt], a.hasOwnProperty(Nt) && H !== void 0 && !i.hasOwnProperty(Nt) && zc(
                e,
                t,
                Nt,
                void 0,
                i,
                H
              );
            for (B in i)
              H = i[B], X = a[B], !i.hasOwnProperty(B) || H === X || H === void 0 && X === void 0 || zc(
                e,
                t,
                B,
                H,
                i,
                X
              );
            return;
          }
      }
      for (var ft in a)
        H = a[ft], a.hasOwnProperty(ft) && H != null && !i.hasOwnProperty(ft) && ht(e, t, ft, null, i, H);
      for (L in i)
        H = i[L], X = a[L], !i.hasOwnProperty(L) || H === X || H == null && X == null || ht(e, t, L, H, i, X);
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
    function Mc(e) {
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
            d != null && typeof d != "boolean" && d !== "" && (f.indexOf("--") === 0 ? (I(d, f), i += o + f + ":" + ("" + d).trim()) : typeof d != "number" || d === 0 || jr.has(f) ? (I(d, f), i += o + f.replace(Zu, "-$1").toLowerCase().replace(Ku, "-ms-") + ":" + ("" + d).trim()) : i += o + f.replace(Zu, "-$1").toLowerCase().replace(Ku, "-ms-") + ":" + d + "px", o = ";");
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
            if (K(i, t), e === "" + i)
              return;
        }
      Bt(t, e, i, f);
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
      Bt(t, e, i, f);
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
            if (K(i, a), e === "" + i)
              return;
        }
      Bt(t, e, i, f);
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
            if (!isNaN(i) && (K(i, t), e === "" + i))
              return;
        }
      Bt(t, e, i, f);
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
            if (K(i, t), a = oo("" + i), e === a)
              return;
        }
      Bt(t, e, i, f);
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
                    typeof b != "string" && typeof b != "number" || Bt(
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
                    d = e.innerHTML, b = b ? b.__html : void 0, b != null && (b = Md(e, b), Bt(
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
                    f.delete("class"), d = Le(
                      e,
                      "class",
                      b
                    ), Bt(
                      "className",
                      d,
                      b,
                      o
                    );
                    continue;
                  default:
                    i.context === kc && t !== "svg" && t !== "math" ? f.delete(v.toLowerCase()) : f.delete(v), d = Le(
                      e,
                      v,
                      b
                    ), Bt(
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
                  typeof v != "string" && typeof v != "number" || Bt(
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
                  f.delete(b), am(e, v, o);
                  continue;
                case "multiple":
                  f.delete(b), Bt(
                    b,
                    e.multiple,
                    v,
                    o
                  );
                  continue;
                case "muted":
                  f.delete(b), Bt(
                    b,
                    e.muted,
                    v,
                    o
                  );
                  continue;
                case "autoFocus":
                  f.delete("autofocus"), Bt(
                    b,
                    e.autofocus,
                    v,
                    o
                  );
                  continue;
                case "data":
                  if (t !== "object") {
                    f.delete(b), d = e.getAttribute("data"), Bt(
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
                    f.delete(b.toLowerCase()), Bt(
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
                    var B = d = b, L = o;
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
                          if (K(v, d), h === "" + v)
                            break e;
                      }
                    Bt(
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
                    if (h = e, B = d = b, L = o, f.delete(B), h = h.getAttribute(B), h === null)
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
                          if (!(isNaN(v) || 1 > v) && (K(v, d), h === "" + v))
                            break e;
                      }
                    Bt(
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
                    h = vs(b), d = !1, i.context === kc && t !== "svg" && t !== "math" ? f.delete(h.toLowerCase()) : (B = b.toLowerCase(), B = Yc.hasOwnProperty(
                      B
                    ) && Yc[B] || null, B !== null && B !== b && (d = !0, f.delete(B)), f.delete(h));
                    e: if (B = e, L = h, h = v, we(L))
                      if (B.hasAttribute(L))
                        B = B.getAttribute(
                          L
                        ), K(
                          h,
                          L
                        ), h = B === "" + h ? h : B;
                      else {
                        switch (typeof h) {
                          case "function":
                          case "symbol":
                            break e;
                          case "boolean":
                            if (B = L.toLowerCase().slice(0, 5), B !== "data-" && B !== "aria-")
                              break e;
                        }
                        h = h === void 0 ? void 0 : null;
                      }
                    else h = void 0;
                    d || Bt(
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
    function at(e, t) {
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
    function nt(e) {
      return e.nodeType === 9 ? e : e.ownerDocument;
    }
    function St(e) {
      switch (e) {
        case af:
          return Uh;
        case Gr:
          return rg;
        default:
          return kc;
      }
    }
    function ya(e, t) {
      if (e === kc)
        switch (t) {
          case "svg":
            return Uh;
          case "math":
            return rg;
          default:
            return kc;
        }
      return e === Uh && t === "foreignObject" ? kc : e;
    }
    function eu(e, t) {
      return e === "textarea" || e === "noscript" || typeof t.children == "string" || typeof t.children == "number" || typeof t.children == "bigint" || typeof t.dangerouslySetInnerHTML == "object" && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null;
    }
    function qo() {
      var e = window.event;
      return e && e.type === "popstate" ? e === q0 ? !1 : (q0 = e, !0) : (q0 = null, !1);
    }
    function im(e) {
      setTimeout(function() {
        throw e;
      });
    }
    function Bu(e, t, a) {
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
    function Yu(e) {
      Fi(e, "");
    }
    function Uc(e, t, a) {
      e.nodeValue = a;
    }
    function tu(e) {
      return e === "head";
    }
    function Fa(e, t) {
      e.removeChild(t);
    }
    function Bo(e, t) {
      (e.nodeType === 9 ? e.body : e.nodeName === "HTML" ? e.ownerDocument.body : e).removeChild(t);
    }
    function Yo(e, t) {
      var a = t, i = 0, o = 0;
      do {
        var f = a.nextSibling;
        if (e.removeChild(a), f && f.nodeType === 8)
          if (a = f.data, a === fg) {
            if (0 < i && 8 > i) {
              a = i;
              var d = e.ownerDocument;
              if (a & iT && Vo(d.documentElement), a & cT && Vo(d.body), a & oT)
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
            a === og || a === Jc || a === pp ? o++ : i = a.charCodeAt(0) - 48;
        else i = 0;
        a = f;
      } while (a);
      Hc(t);
    }
    function ma(e) {
      e = e.style, typeof e.setProperty == "function" ? e.setProperty("display", "none", "important") : e.display = "none";
    }
    function cm(e) {
      e.nodeValue = "";
    }
    function om(e, t) {
      t = t[fT], t = t != null && t.hasOwnProperty("display") ? t.display : null, e.style.display = t == null || typeof t == "boolean" ? "" : ("" + t).trim();
    }
    function Ud(e, t) {
      e.nodeValue = t;
    }
    function jo(e) {
      var t = e.firstChild;
      for (t && t.nodeType === 10 && (t = t.nextSibling); t; ) {
        var a = t;
        switch (t = t.nextSibling, a.nodeName) {
          case "HTML":
          case "HEAD":
          case "BODY":
            jo(a), nn(a);
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
          K(o.name, "name");
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
    function lu(e) {
      return e.data === pp || e.data === Jc && e.ownerDocument.readyState === Tb;
    }
    function Go(e, t) {
      var a = e.ownerDocument;
      if (e.data !== Jc || a.readyState === Tb)
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
          if (t = e.data, t === og || t === pp || t === Jc || t === H0 || t === Sb)
            break;
          if (t === fg) return null;
        }
      }
      return e;
    }
    function _d(e) {
      if (e.nodeType === 1) {
        for (var t = e.nodeName.toLowerCase(), a = {}, i = e.attributes, o = 0; o < i.length; o++) {
          var f = i[o];
          a[lm(f.name)] = f.name.toLowerCase() === "style" ? Mc(e) : f.value;
        }
        return { type: t, props: a };
      }
      return e.nodeType === 8 ? { type: "Suspense", props: {} } : e.nodeValue;
    }
    function Cd(e, t, a) {
      return a === null || a[uT] !== !0 ? (e.nodeValue === t ? e = null : (t = xl(t), e = xl(e.nodeValue) === t ? null : e.nodeValue), e) : null;
    }
    function fm(e) {
      e = e.nextSibling;
      for (var t = 0; e; ) {
        if (e.nodeType === 8) {
          var a = e.data;
          if (a === fg) {
            if (t === 0)
              return Nl(e.nextSibling);
            t--;
          } else
            a !== og && a !== pp && a !== Jc || t++;
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
          if (a === og || a === pp || a === Jc) {
            if (t === 0) return e;
            t--;
          } else a === fg && t++;
        }
        e = e.previousSibling;
      }
      return null;
    }
    function rm(e) {
      Hc(e);
    }
    function Ua(e) {
      Hc(e);
    }
    function sm(e, t, a, i, o) {
      switch (o && ps(e, i.ancestorInfo), t = nt(a), e) {
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
    function _a(e, t, a, i) {
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
    function br(e) {
      return typeof e.getRootNode == "function" ? e.getRootNode() : e.nodeType === 9 ? e : e.ownerDocument;
    }
    function hv(e, t, a) {
      var i = _h;
      if (i && typeof t == "string" && t) {
        var o = Aa(t);
        o = 'link[rel="' + e + '"][href="' + o + '"]', typeof a == "string" && (o += '[crossorigin="' + a + '"]'), zb.has(o) || (zb.add(o), e = { rel: e, crossOrigin: a, href: t }, i.querySelector(o) === null && (t = i.createElement("link"), kt(t, "link", e), D(t), i.head.appendChild(t)));
      }
    }
    function ju(e, t, a, i) {
      var o = (o = nu.current) ? br(o) : null;
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
              state: { loading: ns, preload: null }
            }, f.set(e, d), (f = o.querySelector(
              au(e)
            )) && !f._p && (d.instance = f, d.state.loading = vp | vu), !gu.has(e))) {
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
              gu.set(e, h), f || yv(
                o,
                e,
                h,
                d.state
              );
            }
            if (t && i === null)
              throw a = `

  - ` + _c(t) + `
  + ` + _c(a), Error(
                "Expected <link> not to update to be updated to a stylesheet with precedence. Check the `rel`, `href`, and `precedence` props of this component. Alternatively, check whether two different <link> components render in the same slot or share the same key." + a
              );
            return d;
          }
          if (t && i !== null)
            throw a = `

  - ` + _c(t) + `
  + ` + _c(a), Error(
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
    function _c(e) {
      var t = 0, a = "<link";
      return typeof e.rel == "string" ? (t++, a += ' rel="' + e.rel + '"') : Vu.call(e, "rel") && (t++, a += ' rel="' + (e.rel === null ? "null" : "invalid type " + typeof e.rel) + '"'), typeof e.href == "string" ? (t++, a += ' href="' + e.href + '"') : Vu.call(e, "href") && (t++, a += ' href="' + (e.href === null ? "null" : "invalid type " + typeof e.href) + '"'), typeof e.precedence == "string" ? (t++, a += ' precedence="' + e.precedence + '"') : Vu.call(e, "precedence") && (t++, a += " precedence={" + (e.precedence === null ? "null" : "invalid type " + typeof e.precedence) + "}"), Object.getOwnPropertyNames(e).length > t && (a += " ..."), a + " />";
    }
    function zi(e) {
      return 'href="' + Aa(e) + '"';
    }
    function au(e) {
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
      }), kt(t, "link", a), D(t), e.head.appendChild(t));
    }
    function Cc(e) {
      return '[src="' + Aa(e) + '"]';
    }
    function xc(e) {
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
              return t.instance = i, D(i), i;
            var o = ke({}, a, {
              "data-href": a.href,
              "data-precedence": a.precedence,
              href: null,
              precedence: null
            });
            return i = (e.ownerDocument || e).createElement("style"), D(i), kt(i, "style", o), Hd(i, a.precedence, e), t.instance = i;
          case "stylesheet":
            o = zi(a.href);
            var f = e.querySelector(
              au(o)
            );
            if (f)
              return t.state.loading |= vu, t.instance = f, D(f), f;
            i = dm(a), (o = gu.get(o)) && hm(i, o), f = (e.ownerDocument || e).createElement("link"), D(f);
            var d = f;
            return d._p = new Promise(function(h, v) {
              d.onload = h, d.onerror = v;
            }), kt(f, "link", i), t.state.loading |= vu, Hd(f, a.precedence, e), t.instance = f;
          case "script":
            return f = Cc(a.src), (o = e.querySelector(
              xc(f)
            )) ? (t.instance = o, D(o), o) : (i = a, (o = gu.get(f)) && (i = ke({}, a), ym(i, o)), e = e.ownerDocument || e, o = e.createElement("script"), D(o), kt(o, "link", i), e.head.appendChild(o), t.instance = o);
          case "void":
            return null;
          default:
            throw Error(
              'acquireResource encountered a resource type it did not expect: "' + t.type + '". this is a bug in React.'
            );
        }
      else
        t.type === "stylesheet" && (t.state.loading & vu) === ns && (i = t.instance, t.state.loading |= vu, Hd(i, a.precedence, e));
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
        if (!(f[Po] || f[Zl] || e === "link" && f.getAttribute("rel") === "stylesheet") && f.namespaceURI !== af) {
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
    function Xo(e, t, a) {
      var i = !a.ancestorInfo.containerTagInScope;
      if (a.context === Uh || t.itemProp != null)
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
              a = [], t.onLoad && a.push("`onLoad`"), o && a.push("`onError`"), f != null && a.push("`disabled`"), o = at(a, "and"), o += a.length === 1 ? " prop" : " props", f = a.length === 1 ? "an " + o : "the " + o, a.length && console.error(
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
      if (t.type === "stylesheet" && (typeof a.media != "string" || matchMedia(a.media).matches !== !1) && (t.state.loading & vu) === ns) {
        if (t.instance === null) {
          var o = zi(a.href), f = e.querySelector(
            au(o)
          );
          if (f) {
            e = f._p, e !== null && typeof e == "object" && typeof e.then == "function" && (i.count++, i = Tr.bind(i), e.then(i, i)), t.state.loading |= vu, t.instance = f, D(f);
            return;
          }
          f = e.ownerDocument || e, a = dm(a), (o = gu.get(o)) && hm(a, o), f = f.createElement("link"), D(f);
          var d = f;
          d._p = new Promise(function(h, v) {
            d.onload = h, d.onerror = v;
          }), kt(f, "link", a), t.instance = f;
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
      if (!(t.state.loading & vu)) {
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
        o = t.instance, d = o.getAttribute("data-precedence"), f = a.get(d) || i, f === i && a.set(Y0, o), a.set(d, o), this.count++, i = Tr.bind(this), o.addEventListener("load", i), o.addEventListener("error", i), f ? f.parentNode.insertBefore(o, f.nextSibling) : (e = e.nodeType === 9 ? e.head : e, e.insertBefore(o, e.firstChild)), t.state.loading |= vu;
      }
    }
    function wd(e, t, a, i, o, f, d, h) {
      for (this.tag = 1, this.containerInfo = e, this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = as, this.callbackNode = this.next = this.pendingContext = this.context = this.cancelPendingCommit = null, this.callbackPriority = 0, this.expirationTimes = eo(-1), this.entangledLanes = this.shellSuspendCounter = this.errorRecoveryDisabledLanes = this.expiredLanes = this.warmLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = eo(0), this.hiddenUpdates = eo(null), this.identifierPrefix = i, this.onUncaughtError = o, this.onCaughtError = f, this.onRecoverableError = d, this.pooledCache = null, this.pooledCacheLanes = 0, this.formState = h, this.incompleteTransitions = /* @__PURE__ */ new Map(), this.passiveEffectDuration = this.effectDuration = -0, this.memoizedUpdaters = /* @__PURE__ */ new Set(), e = this.pendingUpdatersLaneMap = [], t = 0; 31 > t; t++) e.push(/* @__PURE__ */ new Set());
      this._debugRootType = a ? "hydrateRoot()" : "createRoot()";
    }
    function vm(e, t, a, i, o, f, d, h, v, b, B, L) {
      return e = new wd(
        e,
        t,
        a,
        d,
        h,
        v,
        b,
        L
      ), t = wS, f === !0 && (t |= Sa | Ju), Ft && (t |= ta), f = U(3, null, null, t), e.current = f, f.stateNode = e, t = jf(), cc(t), e.pooledCache = t, cc(t), f.memoizedState = {
        element: i,
        isDehydrated: a,
        cache: t
      }, ca(f), e;
    }
    function gm(e) {
      return e ? (e = nf, e) : nf;
    }
    function Et(e, t, a, i, o, f) {
      if (wl && typeof wl.onScheduleFiberRoot == "function")
        try {
          wl.onScheduleFiberRoot(Hi, i, a);
        } catch (d) {
          va || (va = !0, console.error(
            "React instrumentation encountered an error: %s",
            d
          ));
        }
      fe !== null && typeof fe.markRenderScheduled == "function" && fe.markRenderScheduled(t), o = gm(o), i.context === null ? i.context = o : i.pendingContext = o, ba && xa !== null && !Cb && (Cb = !0, console.error(
        `Render methods should be a pure function of props and state; triggering nested component updates from render is not allowed. If necessary, trigger nested updates in componentDidUpdate.

Check the render method of %s.`,
        de(xa) || "Unknown"
      )), i = qn(t), i.payload = { element: a }, f = f === void 0 ? null : f, f !== null && (typeof f != "function" && console.error(
        "Expected the last optional `callback` argument to be a function. Instead received: %s.",
        f
      ), i.callback = f), a = hn(e, i, t), a !== null && (Kt(a, e, t), di(a, e, t));
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
        var t = ia(e, 67108864);
        t !== null && Kt(t, e, 67108864), bm(e, 67108864);
      }
    }
    function xg() {
      return xa;
    }
    function Hg() {
      for (var e = /* @__PURE__ */ new Map(), t = 1, a = 0; 31 > a; a++) {
        var i = Sf(t);
        e.set(t, i), t *= 2;
      }
      return e;
    }
    function Ng(e, t, a, i) {
      var o = j.T;
      j.T = null;
      var f = xe.p;
      try {
        xe.p = ql, Mi(e, t, a, i);
      } finally {
        xe.p = f, j.T = o;
      }
    }
    function Bd(e, t, a, i) {
      var o = j.T;
      j.T = null;
      var f = xe.p;
      try {
        xe.p = En, Mi(e, t, a, i);
      } finally {
        xe.p = f, j.T = o;
      }
    }
    function Mi(e, t, a, i) {
      if (yg) {
        var o = Er(i);
        if (o === null)
          Fl(
            e,
            t,
            i,
            mg,
            a
          ), Ui(e, i);
        else if (Ar(
          o,
          e,
          t,
          a,
          i
        ))
          i.stopPropagation();
        else if (Ui(e, i), t & 4 && -1 < hT.indexOf(e)) {
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
                      $a(f), (At & (qa | Fu)) === An && (Iv = uu() + fb, Dc(0));
                    }
                  }
                  break;
                case 13:
                  h = ia(f, 2), h !== null && Kt(h, f, 2), Rc(), bm(f, 2);
              }
            if (f = Er(i), f === null && Fl(
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
          Fl(
            e,
            t,
            i,
            null,
            a
          );
      }
    }
    function Er(e) {
      return e = Pi(e), Qo(e);
    }
    function Qo(e) {
      if (mg = null, e = ua(e), e !== null) {
        var t = tt(e);
        if (t === null) e = null;
        else {
          var a = t.tag;
          if (a === 13) {
            if (e = Pt(t), e !== null) return e;
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
            case Xd:
              return ql;
            case xr:
              return En;
            case Wo:
            case jg:
              return Qu;
            case Hr:
              return Jd;
            default:
              return Qu;
          }
        default:
          return Qu;
      }
    }
    function Ui(e, t) {
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
          Sp.delete(t.pointerId);
          break;
        case "gotpointercapture":
        case "lostpointercapture":
          Tp.delete(t.pointerId);
      }
    }
    function pa(e, t, a, i, o, f) {
      return e === null || e.nativeEvent !== f ? (e = {
        blockedOn: t,
        domEventName: a,
        eventSystemFlags: i,
        nativeEvent: f,
        targetContainers: [o]
      }, t !== null && (t = zl(t), t !== null && Sm(t)), e) : (e.eventSystemFlags |= i, t = e.targetContainers, o !== null && t.indexOf(o) === -1 && t.push(o), e);
    }
    function Ar(e, t, a, i, o) {
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
          return Sp.set(
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
        case "gotpointercapture":
          return f = o.pointerId, Tp.set(
            f,
            pa(
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
      var t = ua(e.target);
      if (t !== null) {
        var a = tt(t);
        if (a !== null) {
          if (t = a.tag, t === 13) {
            if (t = Pt(a), t !== null) {
              e.blockedOn = t, lo(e.priority, function() {
                if (a.tag === 13) {
                  var i = ha(a);
                  i = Ol(i);
                  var o = ia(
                    a,
                    i
                  );
                  o !== null && Kt(o, a, i), bm(a, i);
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
    function Rr(e) {
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
          return t = zl(a), t !== null && Sm(t), e.blockedOn = a, !1;
        t.shift();
      }
      return !0;
    }
    function Tm(e, t, a) {
      Rr(e) && a.delete(t);
    }
    function Sv() {
      j0 = !1, mf !== null && Rr(mf) && (mf = null), pf !== null && Rr(pf) && (pf = null), vf !== null && Rr(vf) && (vf = null), Sp.forEach(Tm), Tp.forEach(Tm);
    }
    function Or(e, t) {
      e.blockedOn === t && (e.blockedOn = null, j0 || (j0 = !0, Wt.unstable_scheduleCallback(
        Wt.unstable_NormalPriority,
        Sv
      )));
    }
    function Tv(e) {
      pg !== e && (pg = e, Wt.unstable_scheduleCallback(
        Wt.unstable_NormalPriority,
        function() {
          pg === e && (pg = null);
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
        return Or(v, e);
      }
      mf !== null && Or(mf, e), pf !== null && Or(pf, e), vf !== null && Or(vf, e), Sp.forEach(t), Tp.forEach(t);
      for (var a = 0; a < gf.length; a++) {
        var i = gf[a];
        i.blockedOn === e && (i.blockedOn = null);
      }
      for (; 0 < gf.length && (a = gf[0], a.blockedOn === null); )
        bv(a), a.blockedOn === null && gf.shift();
      if (a = (e.ownerDocument || e).$$reactFormReplay, a != null)
        for (i = 0; i < a.length; i += 3) {
          var o = a[i], f = a[i + 1], d = o[ga] || null;
          if (typeof f == "function")
            d || Tv(a);
          else if (d) {
            var h = null;
            if (f && f.hasAttribute("formAction")) {
              if (o = f, d = f[ga] || null)
                h = d.formAction;
              else if (Qo(o) !== null) continue;
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
      e[wi] && (e._reactRootContainer ? console.error(
        "You are calling ReactDOMClient.createRoot() on a container that was previously passed to ReactDOM.render(). This is not supported."
      ) : console.error(
        "You are calling ReactDOMClient.createRoot() on a container that has already been passed to createRoot() before. Instead, call root.render() on the existing root instead if you want to update it."
      ));
    }
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(Error());
    var Wt = cS(), zr = xh(), wg = oS(), ke = Object.assign, Mr = Symbol.for("react.element"), _i = Symbol.for("react.transitional.element"), Nc = Symbol.for("react.portal"), Xe = Symbol.for("react.fragment"), Zo = Symbol.for("react.strict_mode"), Ko = Symbol.for("react.profiler"), Em = Symbol.for("react.provider"), Gd = Symbol.for("react.consumer"), Ia = Symbol.for("react.context"), Gu = Symbol.for("react.forward_ref"), Jo = Symbol.for("react.suspense"), Ci = Symbol.for("react.suspense_list"), Ur = Symbol.for("react.memo"), Ca = Symbol.for("react.lazy"), Am = Symbol.for("react.activity"), Av = Symbol.for("react.memo_cache_sentinel"), Rm = Symbol.iterator, Ld = Symbol.for("react.client.reference"), qe = Array.isArray, j = zr.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, xe = wg.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, qg = Object.freeze({
      pending: !1,
      data: null,
      method: null,
      action: null
    }), _r = [], Cr = [], Pa = -1, Lu = Rt(null), ko = Rt(null), nu = Rt(null), $o = Rt(null), Vu = Object.prototype.hasOwnProperty, Vd = Wt.unstable_scheduleCallback, Bg = Wt.unstable_cancelCallback, Rv = Wt.unstable_shouldYield, Yg = Wt.unstable_requestPaint, uu = Wt.unstable_now, xi = Wt.unstable_getCurrentPriorityLevel, Xd = Wt.unstable_ImmediatePriority, xr = Wt.unstable_UserBlockingPriority, Wo = Wt.unstable_NormalPriority, jg = Wt.unstable_LowPriority, Hr = Wt.unstable_IdlePriority, Gg = Wt.log, Tn = Wt.unstable_setDisableYieldValue, Hi = null, wl = null, fe = null, va = !1, Ft = typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u", Ql = Math.clz32 ? Math.clz32 : Pc, Qd = Math.log, Xu = Math.LN2, Zd = 256, Kd = 4194304, ql = 2, En = 8, Qu = 32, Jd = 268435456, Ni = Math.random().toString(36).slice(2), Zl = "__reactFiber$" + Ni, ga = "__reactProps$" + Ni, wi = "__reactContainer$" + Ni, Om = "__reactEvents$" + Ni, Ov = "__reactListeners$" + Ni, Fo = "__reactHandles$" + Ni, Io = "__reactResources$" + Ni, Po = "__reactMarker$" + Ni, Dv = /* @__PURE__ */ new Set(), en = {}, wc = {}, zv = {
      button: !0,
      checkbox: !0,
      image: !0,
      hidden: !0,
      radio: !0,
      reset: !0,
      submit: !0
    }, kd = RegExp(
      "^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), $d = {}, Wd = {}, qi = 0, Dm, zm, Mv, Mm, ef, Uv, _v;
    cn.__reactDisabledLog = !0;
    var Um, Nr, tf = !1, wr = new (typeof WeakMap == "function" ? WeakMap : Map)(), xa = null, ba = !1, Lg = /[\n"\\]/g, _m = !1, Cm = !1, xm = !1, Hm = !1, Fd = !1, Nm = !1, qr = ["value", "defaultValue"], Cv = !1, xv = /["'&<>\n\t]|^\s|\s$/, wm = "address applet area article aside base basefont bgsound blockquote body br button caption center col colgroup dd details dir div dl dt embed fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hgroup hr html iframe img input isindex li link listing main marquee menu menuitem meta nav noembed noframes noscript object ol p param plaintext pre script section select source style summary table tbody td template textarea tfoot th thead title tr track ul wbr xmp".split(
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
    }, lf = {}, iu = {
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
    }, Zu = /([A-Z])/g, Ku = /^ms-/, Br = /^(?:webkit|moz|o)[A-Z]/, Yr = /^-ms-/, Bi = /-(.)/g, Hv = /;\s*$/, qc = {}, Bc = {}, Nv = !1, Ym = !1, jr = new Set(
      "animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(
        " "
      )
    ), Gr = "http://www.w3.org/1998/Math/MathML", af = "http://www.w3.org/2000/svg", eh = /* @__PURE__ */ new Map([
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
    ]), Yc = {
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
    }, cu = {}, Gm = RegExp(
      "^(aria)-[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), th = RegExp(
      "^(aria)[A-Z][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), Lm = !1, ea = {}, Lr = /^on./, l = /^on[^A-Z]/, n = RegExp(
      "^(aria)-[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), u = RegExp(
      "^(aria)[A-Z][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"
    ), c = /^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i, r = null, s = null, y = null, p = !1, S = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), x = !1;
    if (S)
      try {
        var Z = {};
        Object.defineProperty(Z, "passive", {
          get: function() {
            x = !0;
          }
        }), window.addEventListener("test", Z, Z), window.removeEventListener("test", Z, Z);
      } catch {
        x = !1;
      }
    var $ = null, w = null, Y = null, Ae = {
      eventPhase: 0,
      bubbles: 0,
      cancelable: 0,
      timeStamp: function(e) {
        return e.timeStamp || Date.now();
      },
      defaultPrevented: 0,
      isTrusted: 0
    }, Re = Ul(Ae), yt = ke({}, Ae, { view: 0, detail: 0 }), z = Ul(yt), R, C, J, se = ke({}, yt, {
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
        return "movementX" in e ? e.movementX : (e !== J && (J && e.type === "mousemove" ? (R = e.screenX - J.screenX, C = e.screenY - J.screenY) : C = R = 0, J = e), R);
      },
      movementY: function(e) {
        return "movementY" in e ? e.movementY : C;
      }
    }), Fe = Ul(se), Ee = ke({}, se, { dataTransfer: 0 }), _e = Ul(Ee), El = ke({}, yt, { relatedTarget: 0 }), rt = Ul(El), Yi = ke({}, Ae, {
      animationName: 0,
      elapsedTime: 0,
      pseudoElement: 0
    }), Vg = Ul(Yi), rS = ke({}, Ae, {
      clipboardData: function(e) {
        return "clipboardData" in e ? e.clipboardData : window.clipboardData;
      }
    }), sS = Ul(rS), dS = ke({}, Ae, { data: 0 }), Q0 = Ul(
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
        return e.type === "keypress" ? (e = fo(e), e === 13 ? "Enter" : String.fromCharCode(e)) : e.type === "keydown" || e.type === "keyup" ? mS[e.keyCode] || "Unidentified" : "";
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
        return e.type === "keypress" ? fo(e) : 0;
      },
      keyCode: function(e) {
        return e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
      },
      which: function(e) {
        return e.type === "keypress" ? fo(e) : e.type === "keydown" || e.type === "keyup" ? e.keyCode : 0;
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
    }), TS = Ul(SS), ES = ke({}, Ae, {
      propertyName: 0,
      elapsedTime: 0,
      pseudoElement: 0
    }), AS = Ul(ES), RS = ke({}, se, {
      deltaX: function(e) {
        return "deltaX" in e ? e.deltaX : "wheelDeltaX" in e ? -e.wheelDeltaX : 0;
      },
      deltaY: function(e) {
        return "deltaY" in e ? e.deltaY : "wheelDeltaY" in e ? -e.wheelDeltaY : "wheelDelta" in e ? -e.wheelDelta : 0;
      },
      deltaZ: 0,
      deltaMode: 0
    }), OS = Ul(RS), DS = ke({}, Ae, {
      newState: 0,
      oldState: 0
    }), zS = Ul(DS), MS = [9, 13, 27, 32], K0 = 229, Xg = S && "CompositionEvent" in window, Vm = null;
    S && "documentMode" in document && (Vm = document.documentMode);
    var US = S && "TextEvent" in window && !Vm, J0 = S && (!Xg || Vm && 8 < Vm && 11 >= Vm), k0 = 32, $0 = String.fromCharCode(k0), W0 = !1, lh = !1, _S = {
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
    var Ha = typeof Object.is == "function" ? Object.is : Mg, CS = S && "documentMode" in document && 11 >= document.documentMode, ah = null, Qg = null, Zm = null, Zg = !1, nh = {
      animationend: Ru("Animation", "AnimationEnd"),
      animationiteration: Ru("Animation", "AnimationIteration"),
      animationstart: Ru("Animation", "AnimationStart"),
      transitionrun: Ru("Transition", "TransitionRun"),
      transitionstart: Ru("Transition", "TransitionStart"),
      transitioncancel: Ru("Transition", "TransitionCancel"),
      transitionend: Ru("Transition", "TransitionEnd")
    }, Kg = {}, I0 = {};
    S && (I0 = document.createElement("div").style, "AnimationEvent" in window || (delete nh.animationend.animation, delete nh.animationiteration.animation, delete nh.animationstart.animation), "TransitionEvent" in window || delete nh.transitionend.transition);
    var P0 = lc("animationend"), e1 = lc("animationiteration"), t1 = lc("animationstart"), xS = lc("transitionrun"), HS = lc("transitionstart"), NS = lc("transitioncancel"), l1 = lc("transitionend"), a1 = /* @__PURE__ */ new Map(), Jg = "abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(
      " "
    );
    Jg.push("scrollEnd");
    var kg = /* @__PURE__ */ new WeakMap(), wv = 1, jc = 2, ou = [], uh = 0, $g = 0, nf = {};
    Object.freeze(nf);
    var fu = null, ih = null, Yt = 0, wS = 1, ta = 2, Sa = 8, Ju = 16, n1 = 64, u1 = !1;
    try {
      var i1 = Object.preventExtensions({});
    } catch {
      u1 = !0;
    }
    var ch = [], oh = 0, qv = null, Bv = 0, ru = [], su = 0, Vr = null, Gc = 1, Lc = "", Na = null, nl = null, mt = !1, Vc = !1, du = null, Xr = null, ji = !1, Wg = Error(
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
    var Fg = Rt(null), Ig = Rt(null), f1 = {}, Yv = null, fh = null, rh = !1, YS = typeof AbortController < "u" ? AbortController : function() {
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
    }, jS = Wt.unstable_scheduleCallback, GS = Wt.unstable_NormalPriority, Bl = {
      $$typeof: Ia,
      Consumer: null,
      Provider: null,
      _currentValue: null,
      _currentValue2: null,
      _threadCount: 0,
      _currentRenderer: null,
      _currentRenderer2: null
    }, sh = Wt.unstable_now, r1 = -0, jv = -0, tn = -1.1, Qr = -0, Gv = !1, Lv = !1, Km = null, Pg = 0, Zr = 0, dh = null, s1 = j.S;
    j.S = function(e, t) {
      typeof t == "object" && t !== null && typeof t.then == "function" && Zp(e, t), s1 !== null && s1(e, t);
    };
    var Kr = Rt(null), ku = {
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
    ku.recordUnsafeLifecycleWarnings = function(e, t) {
      Jr.has(e.type) || (typeof t.componentWillMount == "function" && t.componentWillMount.__suppressDeprecationWarning !== !0 && Jm.push(e), e.mode & Sa && typeof t.UNSAFE_componentWillMount == "function" && km.push(e), typeof t.componentWillReceiveProps == "function" && t.componentWillReceiveProps.__suppressDeprecationWarning !== !0 && $m.push(e), e.mode & Sa && typeof t.UNSAFE_componentWillReceiveProps == "function" && Wm.push(e), typeof t.componentWillUpdate == "function" && t.componentWillUpdate.__suppressDeprecationWarning !== !0 && Fm.push(e), e.mode & Sa && typeof t.UNSAFE_componentWillUpdate == "function" && Im.push(e));
    }, ku.flushPendingUnsafeLifecycleWarnings = function() {
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
        var d = k(
          t
        );
        console.error(
          `Using UNSAFE_componentWillMount in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move code with side effects to componentDidMount, and set initial state in the constructor.

Please update the following components: %s`,
          d
        );
      }
      0 < i.size && (d = k(
        i
      ), console.error(
        `Using UNSAFE_componentWillReceiveProps in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* If you're updating state whenever props change, refactor your code to use memoization techniques or move it to static getDerivedStateFromProps. Learn more at: https://react.dev/link/derived-state

Please update the following components: %s`,
        d
      )), 0 < f.size && (d = k(
        f
      ), console.error(
        `Using UNSAFE_componentWillUpdate in strict mode is not recommended and may indicate bugs in your code. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.

Please update the following components: %s`,
        d
      )), 0 < e.size && (d = k(e), console.warn(
        `componentWillMount has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move code with side effects to componentDidMount, and set initial state in the constructor.
* Rename componentWillMount to UNSAFE_componentWillMount to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      )), 0 < a.size && (d = k(
        a
      ), console.warn(
        `componentWillReceiveProps has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* If you're updating state whenever props change, refactor your code to use memoization techniques or move it to static getDerivedStateFromProps. Learn more at: https://react.dev/link/derived-state
* Rename componentWillReceiveProps to UNSAFE_componentWillReceiveProps to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      )), 0 < o.size && (d = k(o), console.warn(
        `componentWillUpdate has been renamed, and is not recommended for use. See https://react.dev/link/unsafe-component-lifecycles for details.

* Move data fetching code or side effects to componentDidUpdate.
* Rename componentWillUpdate to UNSAFE_componentWillUpdate to suppress this warning in non-strict mode. In React 18.x, only the UNSAFE_ name will work. To rename all deprecated lifecycles to their new names, you can run \`npx react-codemod rename-unsafe-lifecycles\` in your project source folder.

Please update the following components: %s`,
        d
      ));
    };
    var Vv = /* @__PURE__ */ new Map(), d1 = /* @__PURE__ */ new Set();
    ku.recordLegacyContextWarning = function(e, t) {
      for (var a = null, i = e; i !== null; )
        i.mode & Sa && (a = i), i = i.return;
      a === null ? console.error(
        "Expected to find a StrictMode component in a strict mode tree. This error is likely caused by a bug in React. Please file an issue."
      ) : !d1.has(e.type) && (i = Vv.get(a), e.type.contextTypes != null || e.type.childContextTypes != null || t !== null && typeof t.getChildContext == "function") && (i === void 0 && (i = [], Vv.set(a, i)), i.push(e));
    }, ku.flushLegacyContextWarning = function() {
      Vv.forEach(function(e) {
        if (e.length !== 0) {
          var t = e[0], a = /* @__PURE__ */ new Set();
          e.forEach(function(o) {
            a.add(de(o) || "Component"), d1.add(o.type);
          });
          var i = k(a);
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
    }, ku.discardPendingWarnings = function() {
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
    }, ep = null, Qv = !1, hu = 0, yu = 1, wa = 2, la = 4, Yl = 8, y1 = 0, m1 = 1, p1 = 2, t0 = 3, uf = !1, v1 = !1, l0 = null, a0 = !1, hh = Rt(null), Zv = Rt(0), yh, g1 = /* @__PURE__ */ new Set(), b1 = /* @__PURE__ */ new Set(), n0 = /* @__PURE__ */ new Set(), S1 = /* @__PURE__ */ new Set(), cf = 0, Be = null, Ct = null, Al = null, Kv = !1, mh = !1, kr = !1, Jv = 0, tp = 0, Xc = null, LS = 0, VS = 25, G = null, mu = null, Qc = -1, lp = !1, kv = {
      readContext: xt,
      use: Yn,
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
    }, u0 = null, T1 = null, i0 = null, E1 = null, Gi = null, $u = null, $v = null;
    u0 = {
      readContext: function(e) {
        return xt(e);
      },
      use: Yn,
      useCallback: function(e, t) {
        return G = "useCallback", We(), Va(t), kf(e, t);
      },
      useContext: function(e) {
        return G = "useContext", We(), xt(e);
      },
      useEffect: function(e, t) {
        return G = "useEffect", We(), Va(t), Ns(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", We(), Va(a), qs(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        G = "useInsertionEffect", We(), Va(t), Ka(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", We(), Va(t), ws(e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", We(), Va(t);
        var a = j.H;
        j.H = Gi;
        try {
          return Bs(e, t);
        } finally {
          j.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", We();
        var i = j.H;
        j.H = Gi;
        try {
          return st(e, t, a);
        } finally {
          j.H = i;
        }
      },
      useRef: function(e) {
        return G = "useRef", We(), Jf(e);
      },
      useState: function(e) {
        G = "useState", We();
        var t = j.H;
        j.H = Gi;
        try {
          return Uu(e);
        } finally {
          j.H = t;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", We();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", We(), Ys(e, t);
      },
      useTransition: function() {
        return G = "useTransition", We(), Vn();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", We(), Mu(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", We(), Xn();
      },
      useFormState: function(e, t) {
        return G = "useFormState", We(), po(), Eo(e, t);
      },
      useActionState: function(e, t) {
        return G = "useActionState", We(), Eo(e, t);
      },
      useOptimistic: function(e) {
        return G = "useOptimistic", We(), pn(e);
      },
      useHostTransitionStatus: ra,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", We(), yc();
      }
    }, T1 = {
      readContext: function(e) {
        return xt(e);
      },
      use: Yn,
      useCallback: function(e, t) {
        return G = "useCallback", ee(), kf(e, t);
      },
      useContext: function(e) {
        return G = "useContext", ee(), xt(e);
      },
      useEffect: function(e, t) {
        return G = "useEffect", ee(), Ns(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", ee(), qs(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        G = "useInsertionEffect", ee(), Ka(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", ee(), ws(e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", ee();
        var a = j.H;
        j.H = Gi;
        try {
          return Bs(e, t);
        } finally {
          j.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", ee();
        var i = j.H;
        j.H = Gi;
        try {
          return st(e, t, a);
        } finally {
          j.H = i;
        }
      },
      useRef: function(e) {
        return G = "useRef", ee(), Jf(e);
      },
      useState: function(e) {
        G = "useState", ee();
        var t = j.H;
        j.H = Gi;
        try {
          return Uu(e);
        } finally {
          j.H = t;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", ee(), Ys(e, t);
      },
      useTransition: function() {
        return G = "useTransition", ee(), Vn();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", ee(), Mu(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", ee(), Xn();
      },
      useActionState: function(e, t) {
        return G = "useActionState", ee(), Eo(e, t);
      },
      useFormState: function(e, t) {
        return G = "useFormState", ee(), po(), Eo(e, t);
      },
      useOptimistic: function(e) {
        return G = "useOptimistic", ee(), pn(e);
      },
      useHostTransitionStatus: ra,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), yc();
      }
    }, i0 = {
      readContext: function(e) {
        return xt(e);
      },
      use: Yn,
      useCallback: function(e, t) {
        return G = "useCallback", ee(), dc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", ee(), xt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", ee(), rl(2048, Yl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", ee(), Ln(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", ee(), rl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", ee(), rl(4, la, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", ee();
        var a = j.H;
        j.H = $u;
        try {
          return vi(e, t);
        } finally {
          j.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", ee();
        var i = j.H;
        j.H = $u;
        try {
          return Qa(e, t, a);
        } finally {
          j.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", ee(), ot().memoizedState;
      },
      useState: function() {
        G = "useState", ee();
        var e = j.H;
        j.H = $u;
        try {
          return Qa(dt);
        } finally {
          j.H = e;
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
        return G = "useId", ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", ee(), po(), Hs(e);
      },
      useActionState: function(e) {
        return G = "useActionState", ee(), Hs(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", ee(), _u(e, t);
      },
      useHostTransitionStatus: ra,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), ot().memoizedState;
      }
    }, E1 = {
      readContext: function(e) {
        return xt(e);
      },
      use: Yn,
      useCallback: function(e, t) {
        return G = "useCallback", ee(), dc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", ee(), xt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", ee(), rl(2048, Yl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", ee(), Ln(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", ee(), rl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", ee(), rl(4, la, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", ee();
        var a = j.H;
        j.H = $v;
        try {
          return vi(e, t);
        } finally {
          j.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", ee();
        var i = j.H;
        j.H = $v;
        try {
          return sc(e, t, a);
        } finally {
          j.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", ee(), ot().memoizedState;
      },
      useState: function() {
        G = "useState", ee();
        var e = j.H;
        j.H = $v;
        try {
          return sc(dt);
        } finally {
          j.H = e;
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
        return G = "useId", ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", ee(), po(), Ao(e);
      },
      useActionState: function(e) {
        return G = "useActionState", ee(), Ao(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", ee(), xs(e, t);
      },
      useHostTransitionStatus: ra,
      useMemoCache: el,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), ot().memoizedState;
      }
    }, Gi = {
      readContext: function(e) {
        return V(), xt(e);
      },
      use: function(e) {
        return N(), Yn(e);
      },
      useCallback: function(e, t) {
        return G = "useCallback", N(), We(), kf(e, t);
      },
      useContext: function(e) {
        return G = "useContext", N(), We(), xt(e);
      },
      useEffect: function(e, t) {
        return G = "useEffect", N(), We(), Ns(e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", N(), We(), qs(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        G = "useInsertionEffect", N(), We(), Ka(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", N(), We(), ws(e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", N(), We();
        var a = j.H;
        j.H = Gi;
        try {
          return Bs(e, t);
        } finally {
          j.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", N(), We();
        var i = j.H;
        j.H = Gi;
        try {
          return st(e, t, a);
        } finally {
          j.H = i;
        }
      },
      useRef: function(e) {
        return G = "useRef", N(), We(), Jf(e);
      },
      useState: function(e) {
        G = "useState", N(), We();
        var t = j.H;
        j.H = Gi;
        try {
          return Uu(e);
        } finally {
          j.H = t;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", N(), We();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", N(), We(), Ys(e, t);
      },
      useTransition: function() {
        return G = "useTransition", N(), We(), Vn();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", N(), We(), Mu(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", N(), We(), Xn();
      },
      useFormState: function(e, t) {
        return G = "useFormState", N(), We(), Eo(e, t);
      },
      useActionState: function(e, t) {
        return G = "useActionState", N(), We(), Eo(e, t);
      },
      useOptimistic: function(e) {
        return G = "useOptimistic", N(), We(), pn(e);
      },
      useMemoCache: function(e) {
        return N(), el(e);
      },
      useHostTransitionStatus: ra,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", We(), yc();
      }
    }, $u = {
      readContext: function(e) {
        return V(), xt(e);
      },
      use: function(e) {
        return N(), Yn(e);
      },
      useCallback: function(e, t) {
        return G = "useCallback", N(), ee(), dc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", N(), ee(), xt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", N(), ee(), rl(2048, Yl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", N(), ee(), Ln(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", N(), ee(), rl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", N(), ee(), rl(4, la, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", N(), ee();
        var a = j.H;
        j.H = $u;
        try {
          return vi(e, t);
        } finally {
          j.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", N(), ee();
        var i = j.H;
        j.H = $u;
        try {
          return Qa(e, t, a);
        } finally {
          j.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", N(), ee(), ot().memoizedState;
      },
      useState: function() {
        G = "useState", N(), ee();
        var e = j.H;
        j.H = $u;
        try {
          return Qa(dt);
        } finally {
          j.H = e;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", N(), ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", N(), ee(), $f(e, t);
      },
      useTransition: function() {
        return G = "useTransition", N(), ee(), Ls();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", N(), ee(), Xf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", N(), ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", N(), ee(), Hs(e);
      },
      useActionState: function(e) {
        return G = "useActionState", N(), ee(), Hs(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", N(), ee(), _u(e, t);
      },
      useMemoCache: function(e) {
        return N(), el(e);
      },
      useHostTransitionStatus: ra,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), ot().memoizedState;
      }
    }, $v = {
      readContext: function(e) {
        return V(), xt(e);
      },
      use: function(e) {
        return N(), Yn(e);
      },
      useCallback: function(e, t) {
        return G = "useCallback", N(), ee(), dc(e, t);
      },
      useContext: function(e) {
        return G = "useContext", N(), ee(), xt(e);
      },
      useEffect: function(e, t) {
        G = "useEffect", N(), ee(), rl(2048, Yl, e, t);
      },
      useImperativeHandle: function(e, t, a) {
        return G = "useImperativeHandle", N(), ee(), Ln(e, t, a);
      },
      useInsertionEffect: function(e, t) {
        return G = "useInsertionEffect", N(), ee(), rl(4, wa, e, t);
      },
      useLayoutEffect: function(e, t) {
        return G = "useLayoutEffect", N(), ee(), rl(4, la, e, t);
      },
      useMemo: function(e, t) {
        G = "useMemo", N(), ee();
        var a = j.H;
        j.H = $u;
        try {
          return vi(e, t);
        } finally {
          j.H = a;
        }
      },
      useReducer: function(e, t, a) {
        G = "useReducer", N(), ee();
        var i = j.H;
        j.H = $u;
        try {
          return sc(e, t, a);
        } finally {
          j.H = i;
        }
      },
      useRef: function() {
        return G = "useRef", N(), ee(), ot().memoizedState;
      },
      useState: function() {
        G = "useState", N(), ee();
        var e = j.H;
        j.H = $u;
        try {
          return sc(dt);
        } finally {
          j.H = e;
        }
      },
      useDebugValue: function() {
        G = "useDebugValue", N(), ee();
      },
      useDeferredValue: function(e, t) {
        return G = "useDeferredValue", N(), ee(), js(e, t);
      },
      useTransition: function() {
        return G = "useTransition", N(), ee(), Vs();
      },
      useSyncExternalStore: function(e, t, a) {
        return G = "useSyncExternalStore", N(), ee(), Xf(
          e,
          t,
          a
        );
      },
      useId: function() {
        return G = "useId", N(), ee(), ot().memoizedState;
      },
      useFormState: function(e) {
        return G = "useFormState", N(), ee(), Ao(e);
      },
      useActionState: function(e) {
        return G = "useActionState", N(), ee(), Ao(e);
      },
      useOptimistic: function(e, t) {
        return G = "useOptimistic", N(), ee(), xs(e, t);
      },
      useMemoCache: function(e) {
        return N(), el(e);
      },
      useHostTransitionStatus: ra,
      useCacheRefresh: function() {
        return G = "useCacheRefresh", ee(), ot().memoizedState;
      }
    };
    var A1 = {
      react_stack_bottom_frame: function(e, t, a) {
        var i = ba;
        ba = !0;
        try {
          return e(t, a);
        } finally {
          ba = i;
        }
      }
    }, c0 = A1.react_stack_bottom_frame.bind(A1), R1 = {
      react_stack_bottom_frame: function(e) {
        var t = ba;
        ba = !0;
        try {
          return e.render();
        } finally {
          ba = t;
        }
      }
    }, O1 = R1.react_stack_bottom_frame.bind(R1), D1 = {
      react_stack_bottom_frame: function(e, t) {
        try {
          t.componentDidMount();
        } catch (a) {
          Ue(e, e.return, a);
        }
      }
    }, o0 = D1.react_stack_bottom_frame.bind(
      D1
    ), z1 = {
      react_stack_bottom_frame: function(e, t, a, i, o) {
        try {
          t.componentDidUpdate(a, i, o);
        } catch (f) {
          Ue(e, e.return, f);
        }
      }
    }, M1 = z1.react_stack_bottom_frame.bind(
      z1
    ), U1 = {
      react_stack_bottom_frame: function(e, t) {
        var a = t.stack;
        e.componentDidCatch(t.value, {
          componentStack: a !== null ? a : ""
        });
      }
    }, XS = U1.react_stack_bottom_frame.bind(
      U1
    ), _1 = {
      react_stack_bottom_frame: function(e, t, a) {
        try {
          a.componentWillUnmount();
        } catch (i) {
          Ue(e, t, i);
        }
      }
    }, C1 = _1.react_stack_bottom_frame.bind(
      _1
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
          Ue(e, t, i);
        }
      }
    }, ZS = H1.react_stack_bottom_frame.bind(H1), N1 = {
      react_stack_bottom_frame: function(e) {
        var t = e._init;
        return t(e._payload);
      }
    }, of = N1.react_stack_bottom_frame.bind(N1), ph = null, ap = 0, Ie = null, f0, w1 = f0 = !1, q1 = {}, B1 = {}, Y1 = {};
    Se = function(e, t, a) {
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
          a != null && e !== a && (i = null, typeof a.tag == "number" ? i = de(a) : typeof a.name == "string" && (i = a.name), i && (d = " It was passed a child from " + i + ".")), ye(t, function() {
            console.error(
              'Each child in a list should have a unique "key" prop.%s%s See https://react.dev/link/warning-keys for more information.',
              f,
              d
            );
          });
        }
      }
    };
    var vh = Pf(!0), j1 = Pf(!1), pu = Rt(null), Li = null, gh = 1, np = 2, jl = Rt(0), G1 = {}, L1 = /* @__PURE__ */ new Set(), V1 = /* @__PURE__ */ new Set(), X1 = /* @__PURE__ */ new Set(), Q1 = /* @__PURE__ */ new Set(), Z1 = /* @__PURE__ */ new Set(), K1 = /* @__PURE__ */ new Set(), J1 = /* @__PURE__ */ new Set(), k1 = /* @__PURE__ */ new Set(), $1 = /* @__PURE__ */ new Set(), W1 = /* @__PURE__ */ new Set();
    Object.freeze(G1);
    var r0 = {
      enqueueSetState: function(e, t, a) {
        e = e._reactInternals;
        var i = ha(e), o = qn(i);
        o.payload = t, a != null && (Sy(a), o.callback = a), t = hn(e, o, i), t !== null && (Kt(t, e, i), di(t, e, i)), Mn(e, i);
      },
      enqueueReplaceState: function(e, t, a) {
        e = e._reactInternals;
        var i = ha(e), o = qn(i);
        o.tag = m1, o.payload = t, a != null && (Sy(a), o.callback = a), t = hn(e, o, i), t !== null && (Kt(t, e, i), di(t, e, i)), Mn(e, i);
      },
      enqueueForceUpdate: function(e, t) {
        e = e._reactInternals;
        var a = ha(e), i = qn(a);
        i.tag = p1, t != null && (Sy(t), i.callback = t), t = hn(e, i, a), t !== null && (Kt(t, e, a), di(t, e, a)), fe !== null && typeof fe.markForceUpdateScheduled == "function" && fe.markForceUpdateScheduled(e, a);
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
    }, bh = null, d0 = null, F1 = Error(
      "This is not a real error. It's an implementation detail of React's selective hydration feature. If this leaks into userspace, it's a bug in React. Please file an issue."
    ), Kl = !1, I1 = {}, P1 = {}, eb = {}, tb = {}, Sh = !1, lb = {}, h0 = {}, y0 = {
      dehydrated: null,
      treeContext: null,
      retryLane: 0,
      hydrationErrors: null
    }, ab = !1, nb = null;
    nb = /* @__PURE__ */ new Set();
    var Zc = !1, hl = !1, m0 = !1, ub = typeof WeakSet == "function" ? WeakSet : Set, Jl = null, Th = null, Eh = null, Rl = null, ln = !1, Wu = null, up = 8192, KS = {
      getCacheForType: function(e) {
        var t = xt(Bl), a = t.data.get(e);
        return a === void 0 && (a = e(), t.data.set(e, a)), a;
      },
      getOwner: function() {
        return xa;
      }
    };
    if (typeof Symbol == "function" && Symbol.for) {
      var ip = Symbol.for;
      ip("selector.component"), ip("selector.has_pseudo_class"), ip("selector.role"), ip("selector.test_id"), ip("selector.text");
    }
    var JS = [], kS = typeof WeakMap == "function" ? WeakMap : Map, An = 0, qa = 2, Fu = 4, Kc = 0, cp = 1, Ah = 2, p0 = 3, $r = 4, Wv = 6, ib = 5, At = An, Ht = null, ut = null, it = 0, an = 0, op = 1, Wr = 2, fp = 3, cb = 4, v0 = 5, Rh = 6, rp = 7, g0 = 8, Fr = 9, zt = an, Rn = null, ff = !1, Oh = !1, b0 = !1, Vi = 0, ul = Kc, rf = 0, sf = 0, S0 = 0, On = 0, Ir = 0, sp = null, Ba = null, Fv = !1, T0 = 0, ob = 300, Iv = 1 / 0, fb = 500, dp = null, df = null, $S = 0, WS = 1, FS = 2, Pr = 0, rb = 1, sb = 2, db = 3, IS = 4, E0 = 5, aa = 0, hf = null, Dh = null, yf = 0, A0 = 0, R0 = null, hb = null, PS = 50, hp = 0, O0 = null, D0 = !1, Pv = !1, eT = 50, es = 0, yp = null, zh = !1, eg = null, yb = !1, mb = /* @__PURE__ */ new Set(), tT = {}, tg = null, Mh = null, z0 = !1, M0 = !1, lg = !1, U0 = !1, ts = 0, _0 = {};
    (function() {
      for (var e = 0; e < Jg.length; e++) {
        var t = Jg[e], a = t.toLowerCase();
        t = t[0].toUpperCase() + t.slice(1), on(a, "on" + t);
      }
      on(P0, "onAnimationEnd"), on(e1, "onAnimationIteration"), on(t1, "onAnimationStart"), on("dblclick", "onDoubleClick"), on("focusin", "onFocus"), on("focusout", "onBlur"), on(xS, "onTransitionRun"), on(HS, "onTransitionStart"), on(NS, "onTransitionCancel"), on(l1, "onTransitionEnd");
    })(), ue("onMouseEnter", ["mouseout", "mouseover"]), ue("onMouseLeave", ["mouseout", "mouseover"]), ue("onPointerEnter", ["pointerout", "pointerover"]), ue("onPointerLeave", ["pointerout", "pointerover"]), te(
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
    ), ag = "_reactListening" + Math.random().toString(36).slice(2), pb = !1, vb = !1, ng = !1, gb = !1, ug = !1, ig = !1, bb = !1, cg = {}, lT = /\r\n?/g, aT = /\u0000|\uFFFD/g, ls = "http://www.w3.org/1999/xlink", x0 = "http://www.w3.org/XML/1998/namespace", nT = "javascript:throw new Error('React form unexpectedly submitted.')", uT = "suppressHydrationWarning", og = "$", fg = "/$", Jc = "$?", pp = "$!", iT = 1, cT = 2, oT = 4, H0 = "F!", Sb = "F", Tb = "complete", fT = "style", kc = 0, Uh = 1, rg = 2, N0 = null, w0 = null, Eb = { dialog: !0, webview: !0 }, q0 = null, Ab = typeof setTimeout == "function" ? setTimeout : void 0, rT = typeof clearTimeout == "function" ? clearTimeout : void 0, as = -1, Rb = typeof Promise == "function" ? Promise : void 0, sT = typeof queueMicrotask == "function" ? queueMicrotask : typeof Rb < "u" ? function(e) {
      return Rb.resolve(null).then(e).catch(im);
    } : Ab, B0 = null, ns = 0, vp = 1, Ob = 2, Db = 3, vu = 4, gu = /* @__PURE__ */ new Map(), zb = /* @__PURE__ */ new Set(), $c = xe.d;
    xe.d = {
      f: function() {
        var e = $c.f(), t = Rc();
        return e || t;
      },
      r: function(e) {
        var t = zl(e);
        t !== null && t.tag === 5 && t.type === "form" ? vy(t) : $c.r(e);
      },
      D: function(e) {
        $c.D(e), hv("dns-prefetch", e, null);
      },
      C: function(e, t) {
        $c.C(e, t), hv("preconnect", e, t);
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
          gu.has(f) || (e = ke(
            {
              rel: "preload",
              href: t === "image" && a && a.imageSrcSet ? void 0 : e,
              as: t
            },
            a
          ), gu.set(f, e), i.querySelector(o) !== null || t === "style" && i.querySelector(
            au(f)
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
          if (!gu.has(f) && (e = ke({ rel: "modulepreload", href: e }, t), gu.set(f, e), a.querySelector(o) === null)) {
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
          ), f || (e = ke({ src: e, async: !0 }, t), (t = gu.get(o)) && ym(e, t), f = a.createElement("script"), D(f), kt(f, "link", e), a.head.appendChild(f)), f = {
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
            var h = { loading: ns, preload: null };
            if (d = i.querySelector(
              au(f)
            ))
              h.loading = vp | vu;
            else {
              e = ke(
                {
                  rel: "stylesheet",
                  href: e,
                  "data-precedence": t
                },
                a
              ), (a = gu.get(f)) && hm(e, a);
              var v = d = i.createElement("link");
              D(v), kt(v, "link", e), v._p = new Promise(function(b, B) {
                v.onload = b, v.onerror = B;
              }), v.addEventListener("load", function() {
                h.loading |= vp;
              }), v.addEventListener("error", function() {
                h.loading |= Ob;
              }), h.loading |= vu, Hd(d, t, i);
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
          ), f || (e = ke({ src: e, async: !0, type: "module" }, t), (t = gu.get(o)) && ym(e, t), f = a.createElement("script"), D(f), kt(f, "link", e), a.head.appendChild(f)), f = {
            type: "script",
            instance: f,
            count: 1,
            state: null
          }, i.set(o, f));
        }
      }
    };
    var _h = typeof document > "u" ? null : document, sg = null, gp = null, Y0 = null, dg = null, us = qg, bp = {
      $$typeof: Ia,
      Provider: null,
      Consumer: null,
      _currentValue: us,
      _currentValue2: us,
      _threadCount: 0
    }, Mb = "%c%s%c ", Ub = "background: #e6e6e6;background: light-dark(rgba(0,0,0,0.1), rgba(255,255,255,0.25));color: #000000;color: light-dark(#000000, #ffffff);border-radius: 2px", _b = "", hg = " ", dT = Function.prototype.bind, Cb = !1, xb = null, Hb = null, Nb = null, wb = null, qb = null, Bb = null, Yb = null, jb = null, Gb = null;
    xb = function(e, t, a, i) {
      t = M(e, t), t !== null && (a = F(t.memoizedState, a, 0, i), t.memoizedState = a, t.baseState = a, e.memoizedProps = ke({}, e.memoizedProps), a = ia(e, 2), a !== null && Kt(a, e, 2));
    }, Hb = function(e, t, a) {
      t = M(e, t), t !== null && (a = ie(t.memoizedState, a, 0), t.memoizedState = a, t.baseState = a, e.memoizedProps = ke({}, e.memoizedProps), a = ia(e, 2), a !== null && Kt(a, e, 2));
    }, Nb = function(e, t, a, i) {
      t = M(e, t), t !== null && (a = re(t.memoizedState, a, i), t.memoizedState = a, t.baseState = a, e.memoizedProps = ke({}, e.memoizedProps), a = ia(e, 2), a !== null && Kt(a, e, 2));
    }, wb = function(e, t, a) {
      e.pendingProps = F(e.memoizedProps, t, 0, a), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ia(e, 2), t !== null && Kt(t, e, 2);
    }, qb = function(e, t) {
      e.pendingProps = ie(e.memoizedProps, t, 0), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ia(e, 2), t !== null && Kt(t, e, 2);
    }, Bb = function(e, t, a) {
      e.pendingProps = re(
        e.memoizedProps,
        t,
        a
      ), e.alternate && (e.alternate.pendingProps = e.pendingProps), t = ia(e, 2), t !== null && Kt(t, e, 2);
    }, Yb = function(e) {
      var t = ia(e, 2);
      t !== null && Kt(t, e, 2);
    }, jb = function(e) {
      Oe = e;
    }, Gb = function(e) {
      he = e;
    };
    var yg = !0, mg = null, j0 = !1, mf = null, pf = null, vf = null, Sp = /* @__PURE__ */ new Map(), Tp = /* @__PURE__ */ new Map(), gf = [], hT = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(
      " "
    ), pg = null;
    if (Dr.prototype.render = jd.prototype.render = function(e) {
      var t = this._internalRoot;
      if (t === null) throw Error("Cannot update an unmounted root.");
      var a = arguments;
      typeof a[1] == "function" ? console.error(
        "does not support the second callback argument. To execute a side effect after rendering, declare it in a component body with useEffect()."
      ) : $e(a[1]) ? console.error(
        "You passed a container to the second argument of root.render(...). You don't need to pass it again since you already passed it to create the root."
      ) : typeof a[1] < "u" && console.error(
        "You passed a second argument to root.render(...) but it only accepts one argument."
      ), a = e;
      var i = t.current, o = ha(i);
      Et(i, o, a, t, null, null);
    }, Dr.prototype.unmount = jd.prototype.unmount = function() {
      var e = arguments;
      if (typeof e[0] == "function" && console.error(
        "does not support a callback argument. To execute a side effect after rendering, declare it in a component body with useEffect()."
      ), e = this._internalRoot, e !== null) {
        this._internalRoot = null;
        var t = e.containerInfo;
        (At & (qa | Fu)) !== An && console.error(
          "Attempted to synchronously unmount a root while React was already rendering. React cannot finish unmounting the root until the current render has completed, which may lead to a race condition."
        ), Et(e.current, 2, null, e, null, null), Rc(), t[wi] = null;
      }
    }, Dr.prototype.unstable_scheduleHydration = function(e) {
      if (e) {
        var t = Ef();
        e = { blockedOn: null, target: e, priority: t };
        for (var a = 0; a < gf.length && t !== 0 && t < gf[a].priority; a++) ;
        gf.splice(a, 0, e), a === 0 && bv(e);
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
    ), xe.findDOMNode = function(e) {
      var t = e._reactInternals;
      if (t === void 0)
        throw typeof e.render == "function" ? Error("Unable to find node on an unmounted component.") : (e = Object.keys(e).join(","), Error(
          "Argument appears to not be a ReactComponent. Keys: " + e
        ));
      return e = lt(t), e = e !== null ? De(e) : null, e = e === null ? null : e.stateNode, e;
    }, !function() {
      var e = {
        bundleType: 1,
        version: "19.1.1",
        rendererPackageName: "react-dom",
        currentDispatcherRef: j,
        reconcilerVersion: "19.1.1"
      };
      return e.overrideHookState = xb, e.overrideHookStateDeletePath = Hb, e.overrideHookStateRenamePath = Nb, e.overrideProps = wb, e.overridePropsDeletePath = qb, e.overridePropsRenamePath = Bb, e.scheduleUpdate = Yb, e.setErrorHandler = jb, e.setSuspenseHandler = Gb, e.scheduleRefresh = Ye, e.scheduleRoot = ae, e.setRefreshHandler = Mt, e.getCurrentFiber = xg, e.getLaneLabelMap = Hg, e.injectProfilingHooks = il, ze(e);
    }() && S && window.top === window.self && (-1 < navigator.userAgent.indexOf("Chrome") && navigator.userAgent.indexOf("Edge") === -1 || -1 < navigator.userAgent.indexOf("Firefox"))) {
      var Lb = window.location.protocol;
      /^(https?|file):$/.test(Lb) && console.info(
        "%cDownload the React DevTools for a better development experience: https://react.dev/link/react-devtools" + (Lb === "file:" ? `
You might need to use a local HTTP server (instead of file://): https://react.dev/link/react-devtools-faq` : ""),
        "font-weight:bold"
      );
    }
    Op.createRoot = function(e, t) {
      if (!$e(e))
        throw Error("Target container is not a DOM element.");
      Ev(e);
      var a = !1, i = "", o = Ty, f = Fp, d = Ks, h = null;
      return t != null && (t.hydrate ? console.warn(
        "hydrate through createRoot is deprecated. Use ReactDOMClient.hydrateRoot(container, <App />) instead."
      ) : typeof t == "object" && t !== null && t.$$typeof === _i && console.error(
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
      ), e[wi] = t.current, Py(e), new jd(t);
    }, Op.hydrateRoot = function(e, t, a) {
      if (!$e(e))
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
      ), t.context = gm(null), a = t.current, i = ha(a), i = Ol(i), o = qn(i), o.callback = null, hn(a, o, i), a = i, t.current.lanes = a, bu(t, a), $a(t), e[wi] = t.current, Py(e), new Dr(t);
    }, Op.version = "19.1.1", typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(Error());
  }()), Op;
}
var lS;
function _T() {
  if (lS) return bg.exports;
  lS = 1;
  function M() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function")) {
      if (It.env.NODE_ENV !== "production")
        throw new Error("^_^");
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(M);
      } catch (F) {
        console.error(F);
      }
    }
  }
  return It.env.NODE_ENV === "production" ? (M(), bg.exports = MT()) : bg.exports = UT(), bg.exports;
}
var CT = _T();
let fS = Dn.createContext(
  /** @type {any} */
  null
);
function xT() {
  let M = Dn.useContext(fS);
  if (!M) throw new Error("RenderContext not found");
  return M;
}
function HT() {
  return xT().model;
}
function bf(M) {
  let F = HT(), re = Dn.useSyncExternalStore(
    (ie) => (F.on(`change:${M}`, ie), () => F.off(`change:${M}`, ie)),
    () => F.get(M)
  ), _ = Dn.useCallback(
    (ie) => {
      F.set(
        M,
        // @ts-expect-error - TS cannot correctly narrow type
        typeof ie == "function" ? ie(F.get(M)) : ie
      ), F.save_changes();
    },
    [F, M]
  );
  return [re, _];
}
function NT(M) {
  return ({ el: F, model: re, experimental: _ }) => {
    let ie = CT.createRoot(F);
    return ie.render(
      Dn.createElement(
        Dn.StrictMode,
        null,
        Dn.createElement(
          fS.Provider,
          { value: { model: re, experimental: _ } },
          Dn.createElement(M)
        )
      )
    ), () => ie.unmount();
  };
}
const wT = ({
  evalStructure: M,
  selectedEvals: F,
  onEvalToggle: re,
  showPrerequisites: _
}) => {
  const [ie, he] = Dn.useState(
    new Set(M.map((Se) => Se.name))
  ), Oe = (Se) => {
    he((N) => {
      const V = new Set(N);
      return V.has(Se) ? V.delete(Se) : V.add(Se), V;
    });
  };
  return /* @__PURE__ */ ge.jsx("div", { className: "eval-category-view", children: M.map((Se) => /* @__PURE__ */ ge.jsxs("div", { className: "category-section", children: [
    /* @__PURE__ */ ge.jsxs(
      "div",
      {
        className: "category-header",
        onClick: () => Oe(Se.name),
        children: [
          /* @__PURE__ */ ge.jsx("span", { className: "expand-icon", children: ie.has(Se.name) ? "▼" : "▶" }),
          /* @__PURE__ */ ge.jsx("h4", { children: Se.name }),
          /* @__PURE__ */ ge.jsxs("span", { className: "eval-count", children: [
            "(",
            Se.children.length,
            " evals)"
          ] })
        ]
      }
    ),
    ie.has(Se.name) && /* @__PURE__ */ ge.jsx("div", { className: "eval-list", children: Se.children.map((N) => {
      if (!N.eval_metadata) return null;
      const V = N.eval_metadata, le = F.includes(V.name);
      return /* @__PURE__ */ ge.jsx(
        "div",
        {
          className: `eval-item ${le ? "selected" : ""}`,
          onClick: () => re(V.name),
          children: /* @__PURE__ */ ge.jsxs("div", { className: "eval-header", children: [
            /* @__PURE__ */ ge.jsx(
              "input",
              {
                type: "checkbox",
                checked: le,
                onChange: () => re(V.name),
                onClick: (k) => k.stopPropagation()
              }
            ),
            /* @__PURE__ */ ge.jsx("span", { className: "eval-name", children: N.name })
          ] })
        },
        V.name
      );
    }) })
  ] }, Se.name)) });
}, qT = ({
  evaluations: M,
  selectedEvals: F,
  onEvalToggle: re,
  viewMode: _,
  showPrerequisites: ie
}) => {
  if (_ === "category") {
    const he = M.reduce((Oe, Se) => (Oe[Se.category] || (Oe[Se.category] = []), Oe[Se.category].push(Se), Oe), {});
    return /* @__PURE__ */ ge.jsx("div", { className: "eval-list-view category-view", children: Object.entries(he).map(([Oe, Se]) => /* @__PURE__ */ ge.jsxs("div", { className: "category-section", children: [
      /* @__PURE__ */ ge.jsx("h4", { className: "category-title", children: Oe }),
      /* @__PURE__ */ ge.jsx("div", { className: "eval-grid", children: Se.map((N) => {
        const V = F.includes(N.name);
        return /* @__PURE__ */ ge.jsx(
          "div",
          {
            className: `eval-card ${V ? "selected" : ""}`,
            onClick: () => re(N.name),
            children: /* @__PURE__ */ ge.jsxs("div", { className: "eval-header", children: [
              /* @__PURE__ */ ge.jsx(
                "input",
                {
                  type: "checkbox",
                  checked: V,
                  onChange: () => re(N.name),
                  onClick: (le) => le.stopPropagation()
                }
              ),
              /* @__PURE__ */ ge.jsx("span", { className: "eval-name", children: N.name.split("/").pop() })
            ] })
          },
          N.name
        );
      }) })
    ] }, Oe)) });
  }
  return /* @__PURE__ */ ge.jsx("div", { className: "eval-list-view", children: M.map((he) => {
    const Oe = F.includes(he.name);
    return /* @__PURE__ */ ge.jsx(
      "div",
      {
        className: `eval-item ${Oe ? "selected" : ""}`,
        onClick: () => re(he.name),
        children: /* @__PURE__ */ ge.jsxs("div", { className: "eval-header", children: [
          /* @__PURE__ */ ge.jsx(
            "input",
            {
              type: "checkbox",
              checked: Oe,
              onChange: () => re(he.name),
              onClick: (Se) => Se.stopPropagation()
            }
          ),
          /* @__PURE__ */ ge.jsx("span", { className: "eval-name", children: he.name })
        ] })
      },
      he.name
    );
  }) });
}, BT = ({
  categoryFilter: M,
  availableCategories: F,
  onFilterChange: re
}) => {
  const _ = (ie) => {
    const he = M.includes(ie) ? M.filter((Oe) => Oe !== ie) : [...M, ie];
    re({ category: he });
  };
  return /* @__PURE__ */ ge.jsx("div", { className: "filter-panel", children: /* @__PURE__ */ ge.jsxs("div", { className: "filter-section", children: [
    /* @__PURE__ */ ge.jsx("label", { className: "filter-label", children: "Categories:" }),
    /* @__PURE__ */ ge.jsx("div", { className: "checkbox-group", children: F.map((ie) => /* @__PURE__ */ ge.jsxs("label", { className: "checkbox-item", children: [
      /* @__PURE__ */ ge.jsx(
        "input",
        {
          type: "checkbox",
          checked: M.length === 0 || M.includes(ie),
          onChange: () => _(ie)
        }
      ),
      /* @__PURE__ */ ge.jsx("span", { className: "category-label", children: ie.replace("_", " ") })
    ] }, ie)) })
  ] }) });
}, YT = ({
  searchTerm: M,
  onSearchChange: F
}) => /* @__PURE__ */ ge.jsx("div", { className: "search-bar", children: /* @__PURE__ */ ge.jsxs("div", { className: "search-input-container", children: [
  /* @__PURE__ */ ge.jsx("span", { className: "search-icon", children: "🔍" }),
  /* @__PURE__ */ ge.jsx(
    "input",
    {
      type: "text",
      placeholder: "Search evaluations by name, description, or tags...",
      value: M,
      onChange: (re) => F(re.target.value),
      className: "search-input"
    }
  ),
  M && /* @__PURE__ */ ge.jsx(
    "button",
    {
      className: "clear-search",
      onClick: () => F(""),
      children: "✕"
    }
  )
] }) }), jT = ({
  evalData: M,
  selectedEvals: F,
  categoryFilter: re,
  viewMode: _,
  searchTerm: ie,
  showPrerequisites: he,
  onSelectionChange: Oe,
  onFilterChange: Se
}) => {
  var tt, Pt;
  const [N, V] = Dn.useState(ie);
  Dn.useEffect(() => {
    V(ie);
  }, [ie]);
  const le = Dn.useMemo(() => M != null && M.evaluations ? M.evaluations.filter((lt) => {
    var De, bt;
    if (N) {
      const Ge = N.toLowerCase(), Tt = lt.name.toLowerCase().includes(Ge), de = ((De = lt.description) == null ? void 0 : De.toLowerCase().includes(Ge)) || !1, Rt = ((bt = lt.tags) == null ? void 0 : bt.some((Te) => Te.toLowerCase().includes(Ge))) || !1;
      if (!Tt && !de && !Rt)
        return !1;
    }
    return !(re.length > 0 && !re.includes(lt.category));
  }) : [], [M, N, re]), k = Dn.useMemo(() => {
    if (!(M != null && M.categories)) return [];
    const Me = new Set(le.map((lt) => lt.name));
    return M.categories.map((lt) => ({
      ...lt,
      children: lt.children.filter(
        (De) => De.eval_metadata && Me.has(De.eval_metadata.name)
      )
    })).filter((lt) => lt.children.length > 0);
  }, [M, le]), U = (Me) => {
    const lt = F.includes(Me) ? F.filter((De) => De !== Me) : [...F, Me];
    Oe(lt);
  }, ae = () => {
    const Me = le.map((lt) => lt.name);
    Oe(Me);
  }, Ye = () => {
    Oe([]);
  }, Mt = (Me) => {
    Se({
      categoryFilter: Me.category,
      searchTerm: N,
      viewMode: _,
      showPrerequisites: he
    });
  }, $e = (Me) => {
    V(Me), Se({
      categoryFilter: re,
      searchTerm: Me,
      viewMode: _,
      showPrerequisites: he
    });
  };
  return M ? /* @__PURE__ */ ge.jsxs("div", { className: "eval-finder-container", children: [
    /* @__PURE__ */ ge.jsxs("div", { className: "eval-finder-header", children: [
      /* @__PURE__ */ ge.jsx("h3", { children: "🎯 Evaluation Finder" }),
      /* @__PURE__ */ ge.jsxs("div", { className: "stats", children: [
        F.length,
        " of ",
        le.length,
        " selected"
      ] })
    ] }),
    /* @__PURE__ */ ge.jsx(
      YT,
      {
        searchTerm: N,
        onSearchChange: $e
      }
    ),
    /* @__PURE__ */ ge.jsx(
      BT,
      {
        categoryFilter: re,
        availableCategories: ((tt = M.categories) == null ? void 0 : tt.map((Me) => Me.name)) || [],
        onFilterChange: Mt
      }
    ),
    /* @__PURE__ */ ge.jsxs("div", { className: "view-controls", children: [
      /* @__PURE__ */ ge.jsxs("div", { className: "view-mode-selector", children: [
        /* @__PURE__ */ ge.jsx("label", { children: "View:" }),
        /* @__PURE__ */ ge.jsxs(
          "select",
          {
            value: _,
            onChange: (Me) => Se({
              categoryFilter: re,
              searchTerm: N,
              viewMode: Me.target.value,
              showPrerequisites: he
            }),
            children: [
              /* @__PURE__ */ ge.jsx("option", { value: "tree", children: "Tree" }),
              /* @__PURE__ */ ge.jsx("option", { value: "list", children: "List" }),
              /* @__PURE__ */ ge.jsx("option", { value: "category", children: "By Category" })
            ]
          }
        )
      ] }),
      /* @__PURE__ */ ge.jsxs("div", { className: "selection-controls", children: [
        /* @__PURE__ */ ge.jsxs(
          "button",
          {
            onClick: ae,
            className: "btn-select-all",
            disabled: F.length === le.length,
            title: `Select all ${le.length} evaluations`,
            children: [
              "📋 Select All (",
              le.length,
              ")"
            ]
          }
        ),
        /* @__PURE__ */ ge.jsxs(
          "button",
          {
            onClick: Ye,
            className: "btn-clear-all",
            disabled: F.length === 0,
            title: "Clear all selections",
            children: [
              "🗑️ Clear (",
              F.length,
              ")"
            ]
          }
        )
      ] })
    ] }),
    /* @__PURE__ */ ge.jsx("div", { className: "eval-content", children: le.length === 0 ? /* @__PURE__ */ ge.jsxs("div", { className: "no-evals-message", children: [
      /* @__PURE__ */ ge.jsx("h4", { children: "No evaluations found" }),
      ((Pt = M == null ? void 0 : M.evaluations) == null ? void 0 : Pt.length) > 0 ? /* @__PURE__ */ ge.jsx("p", { children: "No evaluations match your current filters. Try adjusting your search or category filters." }) : /* @__PURE__ */ ge.jsx("div", { children: /* @__PURE__ */ ge.jsx("p", { children: "No evaluation data available." }) })
    ] }) : /* @__PURE__ */ ge.jsx(ge.Fragment, { children: _ === "tree" ? /* @__PURE__ */ ge.jsx(
      wT,
      {
        evalStructure: k,
        selectedEvals: F,
        onEvalToggle: U,
        showPrerequisites: he
      }
    ) : /* @__PURE__ */ ge.jsx(
      qT,
      {
        evaluations: le,
        selectedEvals: F,
        onEvalToggle: U,
        viewMode: _,
        showPrerequisites: he
      }
    ) }) })
  ] }) : /* @__PURE__ */ ge.jsx("div", { className: "eval-finder-container", children: /* @__PURE__ */ ge.jsx("div", { className: "loading-message", children: "🔍 Loading evaluations..." }) });
};
function GT() {
  const [M] = bf("eval_data"), [F, re] = bf("selected_evals"), [_, ie] = bf("category_filter"), [he, Oe] = bf("view_mode"), [Se, N] = bf("search_term"), [V, le] = bf("show_prerequisites"), [, k] = bf("selection_changed"), [, U] = bf("filter_changed");
  return /* @__PURE__ */ ge.jsx(
    jT,
    {
      evalData: M,
      selectedEvals: F || [],
      categoryFilter: _ || [],
      viewMode: he || "category",
      searchTerm: Se || "",
      showPrerequisites: V !== !1,
      onSelectionChange: (ae) => {
        re(ae), k({
          selected_evals: ae,
          action: "updated",
          timestamp: Date.now()
        });
      },
      onFilterChange: (ae) => {
        ie(ae.categoryFilter || []), N(ae.searchTerm || ""), Oe(ae.viewMode || "category"), le(ae.showPrerequisites !== !1), U({
          ...ae,
          timestamp: Date.now()
        });
      }
    }
  );
}
const LT = {
  render: NT(GT)
};
export {
  LT as default
};
