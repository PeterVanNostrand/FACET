(function () {
  'use strict';

  function download (svgInfo, filename) {
    window.URL = (window.URL || window.webkitURL);
    var blob = new Blob(svgInfo.source, {type: 'text\/xml'});
    var url = window.URL.createObjectURL(blob);
    var body = document.body;
    var a = document.createElement('a');

    body.appendChild(a);
    a.setAttribute('download', filename + '.svg');
    a.setAttribute('href', url);
    a.style.display = 'none';
    a.click();
    a.parentNode.removeChild(a);

    setTimeout(function() {
      window.URL.revokeObjectURL(url);
    }, 10);
  }

  var prefix = {
    svg: 'http://www.w3.org/2000/svg',
    xhtml: 'http://www.w3.org/1999/xhtml',
    xlink: 'http://www.w3.org/1999/xlink',
    xml: 'http://www.w3.org/XML/1998/namespace',
    xmlns: 'http://www.w3.org/2000/xmlns/',
  };

  function setInlineStyles (svg) {

    // add empty svg element
    var emptySvg = window.document.createElementNS(prefix.svg, 'svg');
    window.document.body.appendChild(emptySvg);
    var emptySvgDeclarationComputed = window.getComputedStyle(emptySvg);

    // hardcode computed css styles inside svg
    var allElements = traverse(svg);
    var i = allElements.length;
    while (i--) {
      explicitlySetStyle(allElements[i]);
    }

    emptySvg.parentNode.removeChild(emptySvg);

    function explicitlySetStyle(element) {
      var cSSStyleDeclarationComputed = window.getComputedStyle(element);
      var i;
      var len;
      var key;
      var value;
      var computedStyleStr = '';

      for (i = 0, len = cSSStyleDeclarationComputed.length; i < len; i++) {
        key = cSSStyleDeclarationComputed[i];
        value = cSSStyleDeclarationComputed.getPropertyValue(key);
        // CUSTOM adding custom application for SVG font size
        if (key == "font-size" && (element.tagName == "text" || element.tagName == "tspan")) {
          d3.select(element).attr("font-size", value);
        }
        if (value !== emptySvgDeclarationComputed.getPropertyValue(key)) {
          // Don't set computed style of width and height. Makes SVG elmements disappear.
          if ((key !== 'height') && (key !== 'width')) {
            computedStyleStr += key + ':' + value + ';';
          }

        }
      }

      element.setAttribute('style', computedStyleStr);
    }

    function traverse(obj) {
      var tree = [];
      tree.push(obj);
      visit(obj);
      function visit(node) {
        if (node && node.hasChildNodes()) {
          var child = node.firstChild;
          while (child) {
            if (child.nodeType === 1 && child.nodeName != 'SCRIPT') {
              tree.push(child);
              visit(child);
            }

            child = child.nextSibling;
          }
        }
      }

      return tree;
    }
  }

  function preprocess (svg) {
    svg.setAttribute('version', '1.1');

    // removing attributes so they aren't doubled up
    svg.removeAttribute('xmlns');
    svg.removeAttribute('xlink');

    // These are needed for the svg
    if (!svg.hasAttributeNS(prefix.xmlns, 'xmlns')) {
      svg.setAttributeNS(prefix.xmlns, 'xmlns', prefix.svg);
    }

    if (!svg.hasAttributeNS(prefix.xmlns, 'xmlns:xlink')) {
      svg.setAttributeNS(prefix.xmlns, 'xmlns:xlink', prefix.xlink);
    }

    setInlineStyles(svg);

    var xmls = new XMLSerializer();
    var source = xmls.serializeToString(svg);
    var doctype = '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">';
    var rect = svg.getBoundingClientRect();
    var svgInfo = {
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      class: svg.getAttribute('class'),
      id: svg.getAttribute('id'),
      childElementCount: svg.childElementCount,
      source: [doctype + source],
    };

    return svgInfo;
  }

  // MODIFIED FROM https://github.com/edeno/d3-save-svg/tree/gh-pages
  // CHANGES TAGGED WITH "CUSTOM"

  function save(svgElement, config) {
      if (svgElement.nodeName !== 'svg' || svgElement.nodeType !== 1) {
          throw 'Need an svg element input';
      }

      var config = config || {};
      var svgInfo = preprocess(svgElement);
      var defaultFileName = getDefaultFileName(svgInfo);
      var filename = config.filename || defaultFileName;
      var svgInfo = preprocess(svgElement);
      download(svgInfo, filename);
  }

  function getDefaultFileName(svgInfo) {
      var defaultFileName = 'untitled';
      if (svgInfo.id) {
          defaultFileName = svgInfo.id;
      } else if (svgInfo.class) {
          defaultFileName = svgInfo.class;
      } else if (window.document.title) {
          defaultFileName = window.document.title.replace(/[^a-z0-9]/gi, '-').toLowerCase();
      }

      return defaultFileName;
  }

  var rect_values = {
      fill: "white",
      round: 5,
      stroke_width: 2,
      stroke: "black",
  };

  var expl_colors = {
      desired: "blue",
      undesired: "red",
      altered_bad: "red",
      altered_good: "blue",
      unaltered: "#939393"
  };

  var ExplanationTypes = Object.freeze({
      Region: "region",
      Example: "example"
  });

  var ExampleGeneration = Object.freeze({
      Nearest: "nearest",
      Centered: "centered"
  });

  var FeatureOrders = Object.freeze({
      Dataset: "dataset",
      LargestChange: "largestchange"
  });

  var FEATURE_ORDER = FeatureOrders.Dataset;
  var OFFSET_UNSCALED = 0.0001;
  var GENERATION_TYPE = ExampleGeneration.Nearest;
  var CURRENCY_DIGITS = 0;

  function wrap(text, width) {
      // function for creating multi line wrapped text taken from the below
      // https://stackoverflow.com/questions/24784302/wrapping-text-in-d3
      var nlines = 1;
      text.each(function () {
          var text = d3.select(this),
              words = text.text().split(/\s+/).reverse(),
              word,
              line = [],
              lineNumber = 0,
              lineHeight = 1.5, // ems
              x = text.attr("x"),
              y = text.attr("y"),
              dy = 0, //parseFloat(text.attr("dy")),
              tspan = text.text(null)
                  .append("tspan")
                  .attr("x", x)
                  .attr("y", y)
                  .attr("dy", dy + "em");
          while (word = words.pop()) {
              line.push(word);
              tspan.text(line.join(" "));
              if (tspan.node().getComputedTextLength() > width) {
                  line.pop();
                  tspan.text(line.join(" "));
                  line = [word];
                  nlines += 1;
                  tspan = text.append("tspan")
                      .attr("x", x)
                      .attr("y", y)
                      .attr("dy", ++lineNumber * lineHeight + dy + "em")
                      .text(word);
              }
          }
      });
      return nlines;
  }

  function unscale(scaled_value, feature_id, dataset_details) {
      var feature_value = scaled_value;
      if (dataset_details["normalized"]) {
          var feature_range = dataset_details["max_values"][feature_id] - dataset_details["min_values"][feature_id];
          var feature_value = (feature_value * feature_range) + dataset_details["min_values"][feature_id];
      }
      return feature_value
  }

  var formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: CURRENCY_DIGITS,
      maximumFractionDigits: CURRENCY_DIGITS,
  });

  function pretty_value(feature_value, feature_name, readable) {
      var value_text = "";
      var trimmed_value = parseFloat(feature_value).toFixed(readable["feature_decimals"][feature_name]);
      if (readable["feature_units"][feature_name] == "$") {
          value_text += formatter.format(trimmed_value);
      }
      else {
          value_text += trimmed_value + " " + readable["feature_units"][feature_name];
      }
      return value_text
  }

  function argsort(arr) {
      var len = arr.length;
      var indices = new Array(len);
      for (var i = 0; i < len; ++i) { indices[i] = i; }
      indices.sort(function (a, b) { return arr[a] < arr[b] ? -1 : arr[a] > arr[b] ? 1 : 0; });
      return indices;
  }

  /** A function which computes and returns the size of change along each feature and a list of indices for largest to smallest change order */
  function feature_dists_order(instance, region) {
      var n_features = Object.keys(instance).length;
      var feature_distances = Array(n_features).fill(0);
      for (var i = 0; i < n_features; i++) {
          var feature_id = "x" + i;
          var feature_value = instance[feature_id];
          var lower_bound = parseFloat(region[feature_id][0]);
          var upper_bound = parseFloat(region[feature_id][1]);

          if (feature_value < lower_bound) {
              feature_distances[i] = (lower_bound - feature_value);
          } else if (feature_value > upper_bound) {
              feature_distances[i] = (feature_value - upper_bound);
          }
      }
      var idx_order;
      if (FEATURE_ORDER == FeatureOrders.Dataset) {
          idx_order = Array.from(Array(n_features).keys());
      } else if (FEATURE_ORDER == FeatureOrders.LargestChange) {
          idx_order = argsort(feature_distances).reverse();
      }
      return [feature_distances, idx_order];
  }

  function create_example(instance_val, lower_val, upper_val, offset) {
      var example_val;
      var region_half_width = (upper_val - lower_val) / 2;
      if (region_half_width < offset || GENERATION_TYPE == ExampleGeneration.Centered) {
          offset = region_half_width; // use half the range width instead
      }

      // if the value is too low
      if (instance_val <= lower_val) {
          example_val = lower_val + offset; // increase it to fall in the range
      }
      // if the value is too hight
      else if (instance_val >= upper_val) {
          example_val = upper_val - offset; // decrease it to fall in the range
      }

      return example_val;
  }

  /* Clamps the given value to the semantic min/max value for that feature. If no meaningful semantic min/max value is availible it uses the dataset min/max values*/
  function clamp_value(value, feature_id, readable, dataset_details) {
      var clampped_value = value;
      var feature_name = dataset_details["feature_names"][feature_id];

      // clamp to the low end
      var feature_min = readable["semantic_min"][feature_name];
      if (!feature_min) { feature_min = dataset_details["min_values"][feature_id]; }
      clampped_value = Math.max(clampped_value, feature_min);

      // clamp to the high end
      var feature_max = readable["semantic_max"][feature_name]; // use the semantic min
      if (!feature_max) { feature_max = dataset_details["max_values"][feature_id]; }
      clampped_value = Math.min(clampped_value, feature_max);

      return clampped_value;
  }

  var sideBar = function () {
      var width;
      var height;
      var x;
      var y;
      var explanation;
      var dataset_details;
      var readable;

      var my = function (selection) {
          // add a sidebar box for details
          selection.append("rect")
              .attr("id", "sidebar")
              .attr("width", width)
              .attr("height", height)
              .attr("fill", "white")
              .attr("x", x)
              .attr("y", y)
              .attr("rx", rect_values.round)
              .attr("ry", rect_values.round)
              .attr("stroke", rect_values.stroke)
              .attr("stroke-width", rect_values.stroke_width)
              .attr("stroke-linejoin", "round");

          // STATUS label
          var text_margin_left = 15;
          var text_margin_top = 40;
          selection.append("text")
              .text("STATUS")
              .attr("class", "header-text")
              .attr("x", x + text_margin_left)
              .attr("y", y + text_margin_top)
              .attr("fill", "black")
              .attr("font-size", 20);

          // dividing line properties
          var line_width = 2;
          var line_text_expand = 5;
          var line_length = width - (text_margin_left * 2) + line_text_expand;
          var line_x = x + text_margin_left - line_text_expand;

          // STATUS line
          var status_line_y = y + text_margin_top + 10;
          selection.append("line")
              .attr("x1", line_x)
              .attr("x2", line_x + line_length)
              .attr("y1", status_line_y)
              .attr("y2", status_line_y)
              .attr("stroke", rect_values.stroke)
              .attr("stroke-width", line_width);

          // status message
          var center = x + text_margin_left + (width - 2 * text_margin_left) / 2;
          var reject_text = selection.append("text")
              .attr("x", center)
              .attr("y", status_line_y + 30)
              .attr("fill", "black")
              .text("Our algorithm has decided that your loan application should be")
              .attr("text-anchor", "middle")
              .attr("class", "feature-details");
          var n_reject_lines = wrap(reject_text, (width - 2 * text_margin_left));

          var predicted_status_y = status_line_y + n_reject_lines * 45;
          selection.append("text")
              .text(readable["scenario_terms"]["undesired_outcome"].toUpperCase())
              .attr("x", center)
              .attr("y", predicted_status_y)
              .attr("id", "predicated-status")
              .attr("text-anchor", "middle");

          // APPLICATION label
          var application_label_y = predicted_status_y + text_margin_top;
          selection.append("text")
              .text(readable["scenario_terms"]["instance_name"].toUpperCase())
              .attr("class", "header-text")
              .attr("x", x + text_margin_left)
              .attr("y", application_label_y)
              .attr("fill", "black")
              .attr("font-size", 20);

          // APPLICATION line
          var features_line_y = application_label_y + 10;
          selection.append("line")
              .attr("x1", line_x)
              .attr("x2", line_x + line_length)
              .attr("y1", features_line_y)
              .attr("y2", features_line_y)
              .attr("stroke", rect_values.stroke)
              .attr("stroke-width", line_width);

          // APPLICATION FEATURE VALUE CONTENT
          var text_offset = 25;
          var instance = explanation["instance"];
          var n_features = Object.keys(instance).length;
          for (var i = 0; i < n_features; i++) {
              // get the raw feature name and its formatted pretty equivalent
              var feature_id = "x" + i; // x0, x1, ...., xn
              var feature_name = dataset_details["feature_names"][feature_id]; // raw name e.g. applicant_income
              var pretty_name = readable["pretty_feature_names"][feature_name]; // formatted name e.g. Applicant Name

              // get the the raw feature value and unscale it if needed
              var feature_value = unscale(instance[feature_id], feature_id, dataset_details);

              // format the value to a string and add unit signs
              var value_text = pretty_value(feature_value, dataset_details["feature_names"][feature_id], readable);

              // display the feature name
              selection.append("text")
                  .text(pretty_name)
                  .attr("class", "feature-details")
                  .attr("x", x + text_margin_left)
                  .attr("y", features_line_y + text_offset * (i + 1))
                  .attr("fill", "black");

              // display the formatted feature value
              selection.append("text")
                  .text(value_text)
                  .attr("class", "feature-details")
                  .attr("x", x + width - text_margin_left)
                  .attr("y", features_line_y + text_offset * (i + 1))
                  .attr("fill", "black")
                  .attr("text-anchor", "end");
          }
      };

      my.width = function (_) {
          return arguments.length
              ? ((width = +_), my)
              : width;
      };

      my.height = function (_) {
          return arguments.length
              ? ((height = +_), my)
              : height;
      };

      my.x = function (_) {
          return arguments.length
              ? ((x = +_), my)
              : x;
      };

      my.y = function (_) {
          return arguments.length
              ? ((y = +_), my)
              : y;
      };

      my.explanation = function (_) {
          return arguments.length ? ((explanation = _), my) : explanation;
      };

      my.dataset_details = function (_) {
          return arguments.length ? ((dataset_details = _), my) : dataset_details;
      };

      my.readable = function (_) {
          return arguments.length ? ((readable = _), my) : readable;
      };

      return my;
  };

  var liguisticDisplay = function () {
      var width;
      var height;
      var explanation;
      var dataset_details;
      var readable;
      var expl_type;

      var my = function (selection) {
          // extract explanation information
          var n_features = Object.keys(dataset_details["feature_names"]).length;
          var instance = explanation["instance"];
          var region = explanation["region"];
          // add a bounding rectangle
          var bbox_width = width - 20;
          var bbox_height = height - 20;
          var bbox_x = 10;
          var bbox_y = 10;
          var bbox_round = 3;
          var bbox_stroke = "black";
          var bbox_stroke_width = 0.5;
          var bbox_stroke_opacity = 0.4;
          selection.append("rect")
              .attr("id", "bbox")
              .attr("width", bbox_width)
              .attr("id", "bbox")
              .attr("height", bbox_height)
              .attr("fill", rect_values.fill)
              .attr("x", bbox_x)
              .attr("y", bbox_y)
              .attr("rx", bbox_round)
              .attr("ry", bbox_round)
              .attr("stroke", bbox_stroke)
              .attr("stroke-width", bbox_stroke_width)
              .attr("stroke-opacity", bbox_stroke_opacity);

          // ####################################### SIDEBAR CONTENT #######################################

          var sidebar_width_ratio = 0.33;
          var sidebar_margin = 10;
          var sidebar_width = bbox_width * sidebar_width_ratio;
          var sidebar_height = bbox_height - (sidebar_margin * 2);
          var sidebar_x = bbox_x + sidebar_margin;
          var sidebar_y = bbox_y + sidebar_margin;
          var sidebar = sideBar()
              .width(sidebar_width)
              .height(sidebar_height)
              .x(sidebar_x)
              .y(sidebar_y)
              .explanation(explanation)
              .dataset_details(dataset_details)
              .readable(readable);
          selection.call(sidebar);

          // ####################################### FUNCTION LIBRARY #######################################

          function get_feature_name(feature_i) {
              var feature_id = "x" + feature_i;
              var feature_name = dataset_details["feature_names"][feature_id];
              var pretty_feature_name = readable["pretty_feature_names"][feature_name];
              return pretty_feature_name
          }

          function get_feature_between(feature_i, region) {
              var feature_id = "x" + feature_i;
              dataset_details["feature_names"][feature_id];
              var lower_value = unscale(region[feature_id][0], feature_id, dataset_details);
              var upper_value = unscale(region[feature_id][1], feature_id, dataset_details);
              lower_value = clamp_value(lower_value, feature_id, readable, dataset_details);
              upper_value = clamp_value(upper_value, feature_id, readable, dataset_details);
              return [lower_value, upper_value]
          }
          function format_case(text) {
              {
                  return text.toUpperCase()
              }
          }

          // ####################################### EXPLANATION BOX #######################################

          // add a containing box for the explanation
          var ebox_width_ratio = 1 - sidebar_width_ratio;
          var ebox_width = (bbox_width * ebox_width_ratio) - 3 * sidebar_margin;
          var ebox_height = bbox_height - (sidebar_margin * 2);
          var ebox_x = bbox_x + (sidebar_width_ratio * bbox_width) + (2 * sidebar_margin);
          var ebox_y = bbox_y + sidebar_margin;
          selection.append("rect")
              .attr("id", "ebox")
              .attr("width", ebox_width)
              .attr("height", ebox_height)
              .attr("fill", "white")
              .attr("x", ebox_x)
              .attr("y", ebox_y)
              .attr("rx", rect_values.round)
              .attr("ry", rect_values.round)
              .attr("stroke", rect_values.stroke)
              .attr("stroke-width", rect_values.stroke_width)
              .attr("stroke-linejoin", "round");

          // ####################################### BUILD EXPLANATION TEXT #######################################

          // compute the unscaled distance along each dimension and sort by it
          var ref = feature_dists_order(instance, region);
          var feature_distances = ref[0];
          var idx_order = ref[1];

          // CREATE THE STARTING TEXT
          var expl_text = Array(); // lets build an array for templated text
          // ADD: Your <instance> would have been <desired outcome> rather than <undesired outcome> if your
          var start_text = "Your " + readable["scenario_terms"]["instance_name"].toLowerCase() + " would have been";
          expl_text.push([start_text, "black"]);
          expl_text.push([format_case(readable["scenario_terms"]["desired_outcome"]), expl_colors.desired]);
          expl_text.push(["rather than", "black"]);
          expl_text.push([format_case(readable["scenario_terms"]["undesired_outcome"]), expl_colors.undesired]);
          expl_text.push(["if your", "black"]);

          // ADD TEXT FOR ALTERED FEATURES
          var n_feats_listed = 0;
          for (var i = 0; i < n_features; i++) {
              // for features in order of largest change
              if (feature_distances[idx_order[i]] > 0) {
                  var feature_id = "x" + idx_order[i];
                  var feature_name = dataset_details["feature_names"][feature_id];
                  if (n_feats_listed > 0) {
                      expl_text.push(["and your", "black"]);
                  }
                  // ADD: <feature name> was between <good value low> and <good value high> rather than <bad value>
                  expl_text.push([format_case(get_feature_name(idx_order[i])) + " was", "black"]);
                  // get the instance and region values
                  var feature_value = unscale(instance[feature_id], feature_id, dataset_details);
                  var ref$1 = get_feature_between(idx_order[i], region);
                  var lower_value = ref$1[0];
                  var upper_value = ref$1[1];

                  if (expl_type == ExplanationTypes.Region) {
                      expl_text.push(["between", "black"]);
                      // format the value text neatly
                      var lower_value_text = pretty_value(lower_value, feature_name, readable);
                      var upper_value_text = pretty_value(upper_value, feature_name, readable);
                      var range_text = lower_value_text + " and " + upper_value_text;
                      expl_text.push([range_text, expl_colors.desired]);
                  }
                  else if (expl_type == ExplanationTypes.Example) {
                      var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                      var example_val = create_example(feature_value, lower_value, upper_value, offset);
                      var example_text = pretty_value(example_val, feature_name, readable);
                      expl_text.push([example_text, expl_colors.desired]);
                  }

                  expl_text.push(["rather than", "black"]);
                  expl_text.push([pretty_value(feature_value, feature_name, readable), expl_colors.undesired]);
                  n_feats_listed += 1; // number of altered features
              }
          }
          // if unaltered feature remain add the below
          if (n_feats_listed < n_features) {
              expl_text.push([", assuming all other features are the same", "black"]);
              // ADD TEXT FOR UNALTERED FEATURES
              var in_paren = false;
              for (var i$1 = 0; i$1 < n_features; i$1++) {
                  if (feature_distances[idx_order[i$1]] == 0) {
                      var feature_text = "";
                      if (!in_paren) {
                          feature_text += "(";
                          in_paren = true;
                      }
                      feature_text += format_case(get_feature_name(idx_order[i$1]));
                      var feature_id$1 = "x" + idx_order[i$1];
                      var feature_name$1 = dataset_details["feature_names"][feature_id$1];
                      var feature_value$1 = unscale(instance[feature_id$1], feature_id$1, dataset_details);
                      expl_text.push([feature_text, "black"]);

                      var current_value_text;
                      if (expl_type == ExplanationTypes.Region) {
                          var ref$2 = get_feature_between(idx_order[i$1], region);
                          var lower_value$1 = ref$2[0];
                          var upper_value$1 = ref$2[1];
                          var lower_value_text$1 = pretty_value(lower_value$1, feature_name$1, readable);
                          var upper_value_text$1 = pretty_value(upper_value$1, feature_name$1, readable);
                          var range_text$1 = lower_value_text$1 + " and " + upper_value_text$1;
                          current_value_text = "between " + range_text$1;
                      }
                      else if (expl_type == ExplanationTypes.Example) {
                          current_value_text = pretty_value(feature_value$1, feature_name$1, readable);
                      }

                      // var current_value_text = get_feature_between(idx_order[i]);
                      if (i$1 < (n_features - 1)) {  // if this is not the last feature add a semicolon
                          current_value_text += ";";
                      }
                      else { // if it is add a closing parenthesis
                          current_value_text += ")";
                      }
                      expl_text.push([current_value_text, "black"]);
                  }
              }

          }
          // ADD TEXT FOR UNALTERED FEATURES
          // for (let i = 0; i < n_features; i++) {
          //     var feature_text = "";
          //     if (!in_paren) {
          //         feature_text += "(";
          //         in_paren = true;
          //     }
          //     feature_text += format_case(get_feature_name(idx_order[i]))
          //     const feature_id = "x" + idx_order[i];
          //     const feature_name = dataset_details["feature_names"][feature_id];
          //     const feature_value = unscale(instance[feature_id], feature_id, dataset_details);
          //     expl_text.push([feature_text, "black"]);

          //     var current_value_text;
          //     if (expl_type == ExplanationTypes.Region) {
          //         const [lower_value, upper_value] = get_feature_between(idx_order[i], region);
          //         const lower_value_text = pretty_value(lower_value, feature_name, readable);
          //         const upper_value_text = pretty_value(upper_value, feature_name, readable);
          //         const range_text = lower_value_text + " and " + upper_value_text;
          //         current_value_text = "between " + range_text;
          //     }
          //     else if (expl_type == ExplanationTypes.Example) {
          //         current_value_text = pretty_value(feature_value, feature_name, readable);
          //     }

          //     // var current_value_text = get_feature_between(idx_order[i]);
          //     if (i < (n_features - 1)) {  // if this is not the last feature add a semicolon
          //         current_value_text += ";"
          //     }
          //     else { // if it is add a closing parenthesis
          //         current_value_text += ")"
          //     }
          //     expl_text.push([current_value_text, "black"]);
          //     i += 1;
          // }

          // ####################################### RENDER EXPLANATION TEXT #######################################

          var combined_text = "";
          for (var i$2 = 0; i$2 < expl_text.length; i$2++) {
              combined_text += expl_text[i$2][0] + " ";
          }

          // text box location
          var etext_margin = 30;
          var etext_x = ebox_x + etext_margin;
          var etext_y = ebox_y + etext_margin + 40;
          var textbox = selection.append("text")
              .attr("x", etext_x)
              .attr("y", etext_y)
              .attr("fill", "black")
              .attr("class", "feature-details")
              .text(null);
          render_textarray(textbox, expl_text, ebox_width - (2 * etext_margin));

          /** Given an array of [<text element>, <color>] tuples, render the svg text with the right color wrapped to the given width adapted from wrap (https://stackoverflow.com/questions/24784302/wrapping-text-in-d3) */
          function render_textarray(textbox, text_arr, width) {
              var lineNumber = 0;
              textbox.text(null); // clear the textbox

              var entry;
              var x = textbox.attr("x");
              var y = textbox.attr("y");
              var dy = 0; //parseFloat(text.attr("dy")),
              var lineHeight = 1.8;
              var line_used_width = 0;
              var tspan;
              text_arr = text_arr.reverse();
              var punctuation = [",", ".", "?", "!", ";", "'", "\""];
              while (entry = text_arr.pop()) { // for each entry in text_arr
                  var entry_words = entry[0].split(/\s+/).reverse();
                  // add a tspan of the right color, include a space betweent words but not punctuation
                  if (line_used_width > 0 && !punctuation.includes(entry_words[entry_words.length - 1])) {
                      line_used_width += 5;
                  }
                  var tspan = textbox.append("tspan")
                      .attr("x", x)
                      .attr("y", y)
                      .attr("dy", lineNumber * lineHeight + dy + "em")
                      .attr("dx", line_used_width)
                      .attr("fill", entry[1]);
                  var word;
                  var line = [];
                  var tspan_width = 0;
                  while (word = entry_words.pop()) {
                      // add the words of the entry to this tspan
                      line.push(word);
                      tspan.text(line.join(" "));
                      // if that tspan gets wider than the acceptable width
                      tspan_width = tspan.node().getComputedTextLength();
                      if ((line_used_width + tspan_width) > width) {
                          line.pop();
                          tspan.text(line.join(" "));
                          // add a new one
                          lineNumber += 1;
                          tspan = textbox.append("tspan")
                              .attr("x", x)
                              .attr("y", y)
                              .attr("dy", lineNumber * lineHeight + dy + "em")
                              .text(word)
                              .attr("fill", entry[1]);
                          tspan_width = tspan.node().getComputedTextLength();
                          line_used_width = 0;
                          line = [word];
                      }
                  }
                  line_used_width += tspan_width;
              }
              return lineNumber;
          }
      };

      my.width = function (_) {
          return arguments.length
              ? ((width = +_), my)
              : width;
      };

      my.height = function (_) {
          return arguments.length
              ? ((height = +_), my)
              : height;
      };

      my.x = function (_) {
          return arguments.length
              ? ((x = +_), my)
              : x;
      };

      my.y = function (_) {
          return arguments.length
              ? ((y = +_), my)
              : y;
      };

      my.explanation = function (_) {
          return arguments.length ? ((explanation = _), my) : explanation;
      };

      my.dataset_details = function (_) {
          return arguments.length ? ((dataset_details = _), my) : dataset_details;
      };

      my.readable = function (_) {
          return arguments.length ? ((readable = _), my) : readable;
      };

      my.expl_type = function (_) {
          return arguments.length ? ((expl_type = _), my) : expl_type;
      };

      my.expl_type = function (_) {
          return arguments.length ? ((expl_type = _), my) : expl_type;
      };

      return my;
  };

  var TableTypes = Object.freeze({
      Application: "application",
      Explanation: "explanation"
  });

  var explanationTable = function () {
      var width;
      var height;
      var x;
      var y;
      var explanation;
      var dataset_details;
      var readable;
      var expl_type;
      var idx_order;
      var feature_distances;
      var table_type;

      var my = function (selection) {
          // table formatting values
          var tr_stroke = "black";
          var tr_stroke_width = 2;
          var tr_text_offset_x = width / 2;
          var tr_text_offset_y = height / 2;
          var table_row_offset = height;
          var tr_pad = 10;
          var font_size = 16;

          var instance = explanation["instance"];
          var region = explanation["region"];
          var n_features = Object.keys(instance).length;

          // add a header above the table
          if (table_type == "application") {
              var header_1 = readable["scenario_terms"]["instance_name"];
              var header_2 = readable["scenario_terms"]["undesired_outcome"];
              var header_2_color = expl_colors.undesired;
          } else if (table_type == "explanation") {
              var header_1 = "Explanation";
              var header_2 = readable["scenario_terms"]["desired_outcome"];
              var header_2_color = expl_colors.desired;
          }
          var etable_header = selection.append("text")
              .attr("x", x + width / 2)
              .attr("y", y - font_size / 2)
              .attr("class", "feature-details")
              .attr("class", "table-header")
              .attr("font-size", 16)
              .attr("font-style", "bold")
              .attr("fill", "black")
              .attr("text-anchor", "middle");
          etable_header.append("tspan")
              .text(header_1 + " (");
          etable_header.append("tspan")
              .text(header_2)
              .attr("fill", header_2_color);
          etable_header.append("tspan")
              .text(")")
              .attr("fill", "black");
          // create the table body
          for (var i = 0; i < n_features; i++) {
              // get the formatted feature name
              var feature_id = "x" + idx_order[i];
              var feature_name = dataset_details["feature_names"][feature_id];
              var has_change = feature_distances[idx_order[i]] > 0;
              var pretty_feature_name = readable["pretty_feature_names"][feature_name];
              var value_text;
              var value_color;
              var label_color;

              if (table_type == "application") {
                  // get the pretty feature value
                  var feature_val = unscale(instance[feature_id], feature_id, dataset_details);
                  value_text = pretty_value(feature_val, feature_name, readable);
                  // color labels and values based on feature difference
                  value_color = has_change ? expl_colors.altered_bad : expl_colors.unaltered;
                  label_color = has_change ? "black" : expl_colors.unaltered;
              }
              else if (table_type == "explanation") {
                  // get the upper and lower bound values
                  var lower_val = unscale(region[feature_id][0], feature_id, dataset_details);
                  var upper_val = unscale(region[feature_id][1], feature_id, dataset_details);
                  lower_val = clamp_value(lower_val, feature_id, readable, dataset_details);
                  upper_val = clamp_value(upper_val, feature_id, readable, dataset_details);

                  if (expl_type == ExplanationTypes.Region) {
                      // format that text neatly
                      var lower_val_text = pretty_value(lower_val, feature_name, readable);
                      var upper_val_text = pretty_value(upper_val, feature_name, readable);
                      value_text = lower_val_text + " - " + upper_val_text;
                  } else if (expl_type == ExplanationTypes.Example) {
                      var feature_value = unscale(instance[feature_id], feature_id, dataset_details);
                      var example_value = feature_value;
                      if (has_change) {
                          var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                          example_value = create_example(feature_value, lower_val, upper_val, offset);
                      }
                      value_text = pretty_value(example_value, feature_name, readable);
                  }

                  // choose label and value colors based on if the feature had to be changed
                  value_color = has_change ? expl_colors.altered_good : expl_colors.unaltered;
                  label_color = has_change ? "black" : expl_colors.unaltered;
              }

              // start a group for each table cell
              var tr = selection.append("g");
              // draw the cell box
              tr.append("rect")
                  .attr("x", x)
                  .attr("y", y + (table_row_offset * i))
                  .attr("width", width)
                  .attr("height", height)
                  .attr("stroke", tr_stroke)
                  .attr("stroke-width", tr_stroke_width);
              // add the feature label to the top-left of the box
              tr.append("text")
                  .text(pretty_feature_name)
                  .attr("x", x + tr_pad)
                  .attr("y", y + (table_row_offset * i) + tr_pad + (font_size / 2))
                  .attr("class", "feature-details")
                  .attr("fill", "black")
                  .attr("text-anchor", "start")
                  .attr("fill", label_color)
                  .attr("font-size", 16);
              // add the feature value to the center of the box
              tr.append("text")
                  .text(value_text)
                  .attr("x", x + tr_text_offset_x)
                  .attr("y", y + (table_row_offset * i) + tr_text_offset_y + font_size / 2)
                  .attr("class", "feature-details")
                  .attr("fill", "black")
                  .attr("text-anchor", "middle")
                  .attr("fill", value_color)
                  .attr("font-size", 16);
          }
      };

      my.width = function (_) {
          return arguments.length
              ? ((width = +_), my)
              : width;
      };

      my.height = function (_) {
          return arguments.length
              ? ((height = +_), my)
              : height;
      };

      my.x = function (_) {
          return arguments.length
              ? ((x = +_), my)
              : x;
      };

      my.y = function (_) {
          return arguments.length
              ? ((y = +_), my)
              : y;
      };

      my.explanation = function (_) {
          return arguments.length ? ((explanation = _), my) : explanation;
      };

      my.dataset_details = function (_) {
          return arguments.length ? ((dataset_details = _), my) : dataset_details;
      };

      my.readable = function (_) {
          return arguments.length ? ((readable = _), my) : readable;
      };

      my.idx_order = function (_) {
          return arguments.length ? ((idx_order = _), my) : idx_order;
      };

      my.feature_distances = function (_) {
          return arguments.length ? ((feature_distances = _), my) : feature_distances;
      };

      my.table_type = function (_) {
          return arguments.length ? ((table_type = _), my) : table_type;
      };

      my.expl_type = function (_) {
          return arguments.length ? ((expl_type = _), my) : expl_type;
      };

      return my;
  };

  var numericDisplay = function () {
      var width;
      var height;
      var explanation;
      var dataset_details;
      var readable;
      var expl_type;

      var my = function (selection) {
          // extract explanation information
          Object.keys(dataset_details["feature_names"]).length;
          var instance = explanation["instance"];
          var region = explanation["region"];

          // add a bounding rectangle
          var bbox_width = width - 20;
          var bbox_height = height - 20;
          var bbox_x = 10;
          var bbox_y = 10;
          var bbox_round = 3;
          var bbox_stroke = "black";
          var bbox_stroke_width = 0.5;
          var bbox_stroke_opacity = 0.4;
          selection.append("rect")
              .attr("id", "bbox")
              .attr("width", bbox_width)
              .attr("id", "bbox")
              .attr("height", bbox_height)
              .attr("fill", rect_values.fill)
              .attr("x", bbox_x)
              .attr("y", bbox_y)
              .attr("rx", bbox_round)
              .attr("ry", bbox_round)
              .attr("stroke", bbox_stroke)
              .attr("stroke-width", bbox_stroke_width)
              .attr("stroke-opacity", bbox_stroke_opacity)
              .attr("filter", "url(#dropshadow)");

          // ####################################### SIDEBAR CONTENT #######################################

          var sidebar_width_ratio = 0.33;
          var sidebar_margin = 10;
          var sidebar_width = bbox_width * sidebar_width_ratio;
          var sidebar_height = bbox_height - (sidebar_margin * 2);
          var sidebar_x = bbox_x + sidebar_margin;
          var sidebar_y = bbox_y + sidebar_margin;
          var sidebar = sideBar()
              .width(sidebar_width)
              .height(sidebar_height)
              .x(sidebar_x)
              .y(sidebar_y)
              .explanation(explanation)
              .dataset_details(dataset_details)
              .readable(readable);
          selection.call(sidebar);


          // ####################################### EXPLANATION CONTENT #######################################

          // add a containing box for the explanation
          var ebox_width_ratio = 1 - sidebar_width_ratio;
          var ebox_width = (bbox_width * ebox_width_ratio) - 3 * sidebar_margin;
          var ebox_height = bbox_height - (sidebar_margin * 2);
          var ebox_x = bbox_x + (sidebar_width_ratio * bbox_width) + (2 * sidebar_margin);
          var ebox_y = bbox_y + sidebar_margin;
          selection.append("rect")
              .attr("id", "ebox")
              .attr("width", ebox_width)
              .attr("height", ebox_height)
              .attr("fill", "white")
              .attr("x", ebox_x)
              .attr("y", ebox_y)
              .attr("rx", rect_values.round)
              .attr("ry", rect_values.round)
              .attr("stroke", rect_values.stroke)
              .attr("stroke-width", rect_values.stroke_width)
              .attr("stroke-linejoin", "round");

          // compute the unscaled distance along each dimension
          var ref = feature_dists_order(instance, region);
          var feature_distances = ref[0];
          var idx_order = ref[1];

          // table shape values
          var tr_width = 200;
          var tr_height = 70;
          var table_margin = (ebox_width - (2 * tr_width)) / 3;

          // APPLICATION table
          var text_margin_top = 40;
          var atable_x_offset = table_margin;
          var atable_y = ebox_y + text_margin_top;
          var atable_x = ebox_x + atable_x_offset;

          var atable = explanationTable()
              .width(tr_width)
              .height(tr_height)
              .x(atable_x)
              .y(atable_y)
              .explanation(explanation)
              .dataset_details(dataset_details)
              .readable(readable)
              .expl_type(expl_type)
              .idx_order(idx_order)
              .feature_distances(feature_distances)
              .table_type(TableTypes.Application);
          selection.call(atable);

          // EXPLANATION table
          var etable_x_offset = (table_margin * 2) + tr_width;
          var etable_y = ebox_y + text_margin_top;
          var etable_x = ebox_x + etable_x_offset;
          var etable = explanationTable()
              .width(tr_width)
              .height(tr_height)
              .x(etable_x)
              .y(etable_y)
              .explanation(explanation)
              .dataset_details(dataset_details)
              .readable(readable)
              .expl_type(expl_type)
              .idx_order(idx_order)
              .feature_distances(feature_distances)
              .table_type(TableTypes.Explanation);
          selection.call(etable);
      };

      my.width = function (_) {
          return arguments.length
              ? ((width = +_), my)
              : width;
      };

      my.height = function (_) {
          return arguments.length
              ? ((height = +_), my)
              : height;
      };

      my.x = function (_) {
          return arguments.length
              ? ((x = +_), my)
              : x;
      };

      my.y = function (_) {
          return arguments.length
              ? ((y = +_), my)
              : y;
      };

      my.explanation = function (_) {
          return arguments.length ? ((explanation = _), my) : explanation;
      };

      my.dataset_details = function (_) {
          return arguments.length ? ((dataset_details = _), my) : dataset_details;
      };

      my.readable = function (_) {
          return arguments.length ? ((readable = _), my) : readable;
      };

      my.expl_type = function (_) {
          return arguments.length ? ((expl_type = _), my) : expl_type;
      };

      return my;
  };

  var visualDisplay = function () {
      var width;
      var height;
      var explanation;
      var dataset_details;
      var readable;
      var expl_type;

      var my = function (selection) {
          // extract explanation information
          var n_features = Object.keys(dataset_details["feature_names"]).length;
          var instance = explanation["instance"];
          var region = explanation["region"];
          // add a bounding rectangle
          var bbox_width = width - 20;
          var bbox_height = height - 20;
          var bbox_x = 10;
          var bbox_y = 10;
          var bbox_round = 3;
          var bbox_stroke = "black";
          var bbox_stroke_width = 0.5;
          var bbox_stroke_opacity = 0.4;
          selection.append("rect")
              .attr("id", "bbox")
              .attr("width", bbox_width)
              .attr("id", "bbox")
              .attr("height", bbox_height)
              .attr("fill", rect_values.fill)
              .attr("x", bbox_x)
              .attr("y", bbox_y)
              .attr("rx", bbox_round)
              .attr("ry", bbox_round)
              .attr("stroke", bbox_stroke)
              .attr("stroke-width", bbox_stroke_width)
              .attr("stroke-opacity", bbox_stroke_opacity);

          // ####################################### SIDEBAR CONTENT #######################################

          var sidebar_width_ratio = 0.33;
          var sidebar_margin = 10;
          var sidebar_width = bbox_width * sidebar_width_ratio;
          var sidebar_height = bbox_height - (sidebar_margin * 2);
          var sidebar_x = bbox_x + sidebar_margin;
          var sidebar_y = bbox_y + sidebar_margin;
          var sidebar = sideBar()
              .width(sidebar_width)
              .height(sidebar_height)
              .x(sidebar_x)
              .y(sidebar_y)
              .explanation(explanation)
              .dataset_details(dataset_details)
              .readable(readable);
          selection.call(sidebar);

          // ####################################### FUNCTION LIBRARY #######################################

          function get_feature_name(feature_i) {
              var feature_id = "x" + feature_i;
              var feature_name = dataset_details["feature_names"][feature_id];
              var pretty_feature_name = readable["pretty_feature_names"][feature_name];
              return pretty_feature_name
          }
          var percent_val = 0.25;

          function get_line_low(value, feature_id) {
              // select the bar min value
              var min_value;
              {
                  min_value = value - (percent_val * value);
              }
              min_value = clamp_value(min_value, feature_id, readable, dataset_details);
              return min_value;
          }

          function get_line_high(value, feature_id) {
              // select the bar max value
              var max_value;
              {
                  max_value = value + (percent_val * value);
              }
              max_value = clamp_value(max_value, feature_id, readable, dataset_details);
              return max_value;
          }

          function pixel_scale(value, min_value, max_value) {
              var clamped_val = value;
              clamped_val = Math.min(clamped_val, max_value);
              clamped_val = Math.max(clamped_val, min_value);
              return ((clamped_val - min_value) / (max_value - min_value)) * line_plot_width;
          }

          // ####################################### EXPLANATION BOX #######################################

          // add a containing box for the explanation
          var ebox_width_ratio = 1 - sidebar_width_ratio;
          var ebox_width = (bbox_width * ebox_width_ratio) - 3 * sidebar_margin;
          var ebox_height = bbox_height - (sidebar_margin * 2);
          var ebox_x = bbox_x + (sidebar_width_ratio * bbox_width) + (2 * sidebar_margin);
          var ebox_y = bbox_y + sidebar_margin;
          selection.append("rect")
              .attr("id", "ebox")
              .attr("width", ebox_width)
              .attr("height", ebox_height)
              .attr("fill", "white")
              .attr("x", ebox_x)
              .attr("y", ebox_y)
              .attr("rx", rect_values.round)
              .attr("ry", rect_values.round)
              .attr("stroke", rect_values.stroke)
              .attr("stroke-width", rect_values.stroke_width)
              .attr("stroke-linejoin", "round");

          // add header bar
          var header_line_y = ebox_y + 50;
          var ebox_padding = 15;
          selection.append("line")
              .attr("x1", ebox_x + ebox_padding)
              .attr("x2", ebox_x + ebox_width - ebox_padding)
              .attr("y1", header_line_y)
              .attr("y2", header_line_y)
              .attr("stroke", rect_values.stroke)
              .attr("stroke-width", 2);

          // Current status (rejected)
          var bad_header = selection.append("text")
              .attr("x", ebox_x + ebox_padding + 5)
              .attr("y", header_line_y - 10)
              .attr("class", "feature-details")
              .attr("class", "table-header")
              .attr("font-size", 16)
              .attr("font-style", "bold")
              .attr("fill", "black");
          bad_header.append("tspan")
              .text(readable["scenario_terms"]["instance_name"] + " (");
          bad_header.append("tspan")
              .text(readable["scenario_terms"]["undesired_outcome"])
              .attr("fill", expl_colors.undesired);
          bad_header.append("tspan")
              .text(")")
              .attr("fill", "black");

          // explanation status (accepted)
          var good_header = selection.append("text")
              .attr("x", ebox_x + ebox_width - ebox_padding - 5)
              .attr("y", header_line_y - 10)
              .attr("class", "feature-details")
              .attr("class", "table-header")
              .attr("font-size", 16)
              .attr("font-style", "bold")
              .attr("fill", "black")
              .attr("text-anchor", "end");
          good_header.append("tspan")
              .text("Explanation (");
          good_header.append("tspan")
              .text(readable["scenario_terms"]["desired_outcome"])
              .attr("fill", expl_colors.desired);
          good_header.append("tspan")
              .text(")")
              .attr("fill", "black");

          // compute the unscaled distance along each dimension
          var ref = feature_dists_order(instance, region);
          var feature_distances = ref[0];
          var idx_order = ref[1];

          var labels_x = ebox_x + 10;
          var labels_width = 150;
          var line_plot_pad_x = 10;
          var line_plot_pad_y = 100;
          var line_plot_x = labels_x + labels_width + line_plot_pad_x;
          var line_plot_width = 290;
          var line_spacing = 55;
          var line_width = 1;
          var tick_height = 10;
          var bar_height = tick_height - 2;
          var circle_radius = bar_height / 2;
          var value_font = 12;

          for (var i = 0; i < n_features; i++) { // for each feature
              var feature_id = "x" + idx_order[i];
              var feature_name = dataset_details["feature_names"][feature_id];

              // ##### DRAW THE NUMBER LINE #####

              // draw a number line for this feature
              var line_y = ebox_y + line_plot_pad_y + line_spacing * i;
              selection.append("line")
                  .attr("x1", line_plot_x)
                  .attr("x2", line_plot_x + line_plot_width)
                  .attr("y1", line_y)
                  .attr("y2", line_y)
                  .attr("stroke", rect_values.stroke)
                  .attr("stroke-width", line_width);

              // add a text label for the line
              selection.append("text")
                  .text(get_feature_name(idx_order[i]) + ":")
                  .attr("x", labels_x + labels_width)
                  .attr("y", line_y + (tick_height / 2))
                  .attr("class", "feature-details")
                  .attr("fill", "black")
                  .attr("text-anchor", "end");
              // add ticks to the ends of the line and label them
              selection.append("line")
                  .attr("x1", line_plot_x)
                  .attr("x2", line_plot_x)
                  .attr("y1", line_y - tick_height / 2)
                  .attr("y2", line_y + tick_height / 2)
                  .attr("stroke", rect_values.stroke)
                  .attr("stroke-width", line_width);
              selection.append("line")
                  .attr("x1", line_plot_x + line_plot_width)
                  .attr("x2", line_plot_x + line_plot_width)
                  .attr("y1", line_y - tick_height / 2)
                  .attr("y2", line_y + tick_height / 2)
                  .attr("stroke", rect_values.stroke)
                  .attr("stroke-width", line_width);


              // determine how to scale the numberline
              var instance_val = unscale(instance[feature_id], feature_id, dataset_details);
              var region_lower = unscale(region[feature_id][0], feature_id, dataset_details);
              var region_upper = unscale(region[feature_id][1], feature_id, dataset_details);

              var has_change = feature_distances[idx_order[i]] > 0;


              var example_val = instance_val;
              if (expl_type == ExplanationTypes.Example) {
                  if (has_change) {
                      var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                      example_val = create_example(instance_val, region_lower, region_upper, offset);
                  }
              }

              var min_plot_value;
              var max_plot_value;
              if (expl_type == ExplanationTypes.Region) {
                  min_plot_value = Math.min(instance_val, region_lower);
                  max_plot_value = Math.max(instance_val, region_upper);
              }
              else if (expl_type == ExplanationTypes.Example) {
                  min_plot_value = Math.min(instance_val, example_val);
                  max_plot_value = Math.max(instance_val, example_val);
                  if (max_plot_value == 0) {
                      max_plot_value = 0.5 * region_upper;
                  }
              }

              var line_min = get_line_low(min_plot_value, feature_id);
              var line_max = get_line_high(max_plot_value, feature_id);

              var bar_lower_val = Math.max(region_lower, line_min);
              var bar_upper_val = Math.min(region_upper, line_max);

              // label the ends of the line
              var line_text_lower = selection.append("text")
                  .text(pretty_value(line_min, feature_name, readable))
                  .attr("font-size", value_font)
                  .attr("fill", "black")
                  .attr("x", line_plot_x)
                  .attr("y", line_y + bar_height + value_font)
                  .attr("text-anchor", "middle")
                  .attr("class", "tick-label");
              selection.append("text")
                  .text(pretty_value(line_max, feature_name, readable))
                  .attr("font-size", value_font)
                  .attr("fill", "black")
                  .attr("x", line_plot_x + line_plot_width)
                  .attr("y", line_y + bar_height + value_font)
                  .attr("text-anchor", "middle")
                  .attr("class", "tick-label");

              // ########## EXPLANTION CONTENT ##########



              // ##### DRAW THE BAR #####
              if (expl_type == ExplanationTypes.Region) {
                  // add a bar for the region range
                  var lower_px = pixel_scale(region_lower, line_min, line_max);
                  var upper_px = pixel_scale(region_upper, line_min, line_max);
                  var bar_color = has_change ? expl_colors.altered_good : expl_colors.unaltered;
                  var bar_width_px = upper_px - lower_px;
                  var bar_start_px = line_plot_x + lower_px;
                  var bar_end_px = bar_start_px + bar_width_px;
                  selection.append("rect")
                      .attr("x", bar_start_px)
                      .attr("y", line_y - (bar_height / 2))
                      .attr("width", bar_width_px, feature_id)
                      .attr("height", bar_height)
                      .attr("fill", bar_color);

                  // add ticks to the ends of the bar
                  selection.append("line")
                      .attr("x1", bar_start_px)
                      .attr("x2", bar_start_px)
                      .attr("y1", line_y - tick_height / 2)
                      .attr("y2", line_y + tick_height / 2)
                      .attr("stroke", rect_values.stroke)
                      .attr("stroke-width", line_width);
                  selection.append("line")
                      .attr("x1", bar_end_px)
                      .attr("x2", bar_end_px)
                      .attr("y1", line_y - tick_height / 2)
                      .attr("y2", line_y + tick_height / 2)
                      .attr("stroke", rect_values.stroke)
                      .attr("stroke-width", line_width);

                  // label the ends of the bar
                  var bar_text_lower = selection.append("text")
                      .text(pretty_value(bar_lower_val, feature_name, readable))
                      .attr("font-size", value_font)
                      .attr("fill", bar_color)
                      .attr("x", bar_start_px)
                      .attr("y", line_y - bar_height)
                      .attr("text-anchor", "end")
                      .attr("class", "tick-label");
                  var bar_text_upper = selection.append("text")
                      .text(pretty_value(bar_upper_val, feature_name, readable))
                      .attr("font-size", value_font)
                      .attr("fill", bar_color)
                      .attr("x", bar_end_px)
                      .attr("y", line_y - bar_height)
                      .attr("text-anchor", "start")
                      .attr("class", "tick-label");

                  // if we have space for the text, center the bar end labels on the ticks
                  var lower_text_width = bar_text_lower.node().getComputedTextLength();
                  var upper_text_width = bar_text_upper.node().getComputedTextLength();
                  if (bar_width_px > ((lower_text_width / 2) + (upper_text_width / 2) + 10)) {
                      bar_text_lower.attr("text-anchor", "middle");
                      bar_text_upper.attr("text-anchor", "middle");
                  }
              }
              // ##### OR DRAW THE EXAMPLE CIRCLE #####
              else if (expl_type == ExplanationTypes.Example) {
                  if (has_change) {
                      var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                      // const example_val = create_example(instance_val, bar_lower_val, bar_upper_val, offset);
                      var expl_circle_x = line_plot_x + pixel_scale(example_val, line_min, line_max);
                      // draw the circle
                      selection.append("circle")
                          .attr("cx", expl_circle_x)
                          .attr("cy", line_y)
                          .attr("r", circle_radius)
                          .attr("fill", expl_colors.altered_good);
                      // add a text label
                      selection.append("text")
                          .text(pretty_value(example_val, feature_name, readable))
                          .attr("font-size", value_font)
                          .attr("fill", expl_colors.altered_good)
                          .attr("x", expl_circle_x)
                          .attr("y", line_y - bar_height)
                          .attr("text-anchor", "middle")
                          .attr("class", "tick-label");
                  }
              }

              // ##### DRAW THE INSTANCE CIRCLE #####

              // add a circle for the instance value
              var unaltered_color = expl_type == ExplanationTypes.Region ? "black" : expl_colors.unaltered;
              var circle_color = has_change ? expl_colors.altered_bad : unaltered_color;
              var circle_x = line_plot_x + pixel_scale(instance_val, line_min, line_max);
              selection.append("circle")
                  .attr("cx", circle_x)
                  .attr("cy", line_y)
                  .attr("r", circle_radius)
                  .attr("fill", circle_color);

              // add text label for instance circle
              var circle_text = selection.append("text")
                  .text(pretty_value(instance_val, feature_name, readable))
                  .attr("font-size", value_font)
                  .attr("fill", circle_color)
                  .attr("x", circle_x)
                  .attr("y", line_y + bar_height + value_font)
                  .attr("text-anchor", "middle")
                  .attr("class", "tick-label");
              // check if the instance's text label will overlap the text label from either end of the line
              if (instance_val != line_min && instance_val != line_max) {
                  // get the size of the rendered text elements
                  var line_lower_text_px = line_text_lower.node().getComputedTextLength() / 2;
                  var line_upper_text_px = line_text_lower.node().getComputedTextLength() / 2;
                  var circle_text_px = circle_text.node().getComputedTextLength() / 2;
                  // compute the start/end of the upper/lower line text labels
                  var line_text_up_start = line_plot_x + line_plot_width - line_upper_text_px;
                  var line_text_low_end = line_plot_x + line_lower_text_px;
                  // compute the size of the overlap
                  var text_gap_up = line_text_up_start - (circle_x + circle_text_px);
                  var text_gap_low = (circle_x - circle_text_px) - line_text_low_end;
                  // if there are fewer than 10 pixels between the text elements, adjust the text position
                  var buffer_size = 10;
                  if (text_gap_up < buffer_size) {
                      var adjusted_x = line_text_up_start - circle_text_px - buffer_size;
                      circle_text.attr("x", adjusted_x);
                  }
                  if (text_gap_low < buffer_size) {
                      var adjusted_x$1 = line_text_low_end + circle_text_px + buffer_size;
                      circle_text.attr("x", adjusted_x$1);
                  }
              }
          }
      };

      my.width = function (_) {
          return arguments.length
              ? ((width = +_), my)
              : width;
      };

      my.height = function (_) {
          return arguments.length
              ? ((height = +_), my)
              : height;
      };

      my.x = function (_) {
          return arguments.length
              ? ((x = +_), my)
              : x;
      };

      my.y = function (_) {
          return arguments.length
              ? ((y = +_), my)
              : y;
      };

      my.explanation = function (_) {
          return arguments.length ? ((explanation = _), my) : explanation;
      };

      my.dataset_details = function (_) {
          return arguments.length ? ((dataset_details = _), my) : dataset_details;
      };

      my.readable = function (_) {
          return arguments.length ? ((readable = _), my) : readable;
      };

      my.expl_type = function (_) {
          return arguments.length ? ((expl_type = _), my) : expl_type;
      };

      return my;
  };

  // group import d3 functions
  var json = d3.json;
  var select = d3.select;

  // data source url
  var detailsURL = "/visualization/data/dataset_details.json";

  var DisplayTypes = Object({
      Numeric: "numeric",
      Linguistic: "linguistic",
      Visual: "visual"
  });

  // main rendering function
  var main = async function () {
      // ####################################### DATA LOADING #######################################

      // load the dataset details
      var dataset_details = await json(detailsURL);

      // load the paths to the explanations JSON.
      var explanation_paths = await json("/visualization/data/explanation_paths.json");

      // load the explanation and isolate the instance and the region. Note we use +/-100000000000000 for +/-Infinity
      var explanation_id = 0;
      var explanation = await json(explanation_paths[explanation_id]);
      var current_expl_type = ExplanationTypes.Region;

      // load human readable info for the dataset
      var readableURL = "/visualization/data/human_readable_details.json";
      var readable = await json(readableURL);

      // ####################################### RENDER EXPLANATION #######################################

      select('#instance-counter')
          .text("Instance " + explanation_id);

      var width = 800;
      var height = 375;
      var svg = select('#svg_container').append("svg")
          .attr('width', width)
          .attr('height', height)
          .attr("fill", "white")
          .attr("id", "image_svg");

      var numeric_display = numericDisplay()
          .width(width)
          .height(height)
          .explanation(explanation)
          .dataset_details(dataset_details)
          .readable(readable)
          .expl_type(current_expl_type);
      var linguistic_display = liguisticDisplay()
          .width(width)
          .height(height)
          .explanation(explanation)
          .dataset_details(dataset_details)
          .readable(readable)
          .expl_type(current_expl_type);
      var visual_display = visualDisplay()
          .width(width)
          .height(height)
          .explanation(explanation)
          .dataset_details(dataset_details)
          .readable(readable)
          .expl_type(current_expl_type);

      var current_display = DisplayTypes.Numeric;
      update_display(current_display);

      // ####################################### BUTTON CONTROLS #######################################

      function update_display(dtype) {
          if (dtype) { current_display = dtype; }
          // clear the SVG and render the new display
          svg.selectAll("*").remove();
          if (current_display == DisplayTypes.Numeric) {
              svg.call(numeric_display);
          } else if (current_display == DisplayTypes.Linguistic) {
              svg.call(linguistic_display);
          }
          else if (current_display == DisplayTypes.Visual) {
              svg.call(visual_display);
          }
      }

      async function update_explantion() {
          explanation = await json(explanation_paths[explanation_id]);
          // update all the displays
          numeric_display.explanation(explanation);
          linguistic_display.explanation(explanation);
          visual_display.explanation(explanation);
          // clear the SVG and render the correct display
          update_display();
      }

      function update_expl_type(etype) {
          current_expl_type = etype;
          // update all the displays
          numeric_display.expl_type(etype);
          linguistic_display.expl_type(etype);
          visual_display.expl_type(etype);
          // clear the SVG and render the correct display
          update_display();
      }

      // numeric button
      d3.select('button#numeric').on('click', function () {
          update_display(DisplayTypes.Numeric);
      });

      // linguistic button
      d3.select('button#linguistic').on('click', function () {
          update_display(DisplayTypes.Linguistic);
      });

      // visual button
      d3.select('button#visual').on('click', function () {
          update_display(DisplayTypes.Visual);
      });

      // example button
      d3.select('button#example').on('click', function () {
          update_expl_type(ExplanationTypes.Example);
      });

      // region button
      d3.select('button#region').on('click', function () {
          update_expl_type(ExplanationTypes.Region);
      });

      function execute_save() {
          var config = {
              filename: 'explanation_' + String(explanation_id).padStart(3, 0) + "_" + current_display + "_" + current_expl_type,
          };
          save(d3.select('svg').node(), config);
      }

      // save button
      d3.select('button#export').on('click', function () {
          execute_save();
      });

      // previous button
      d3.select('button#previous').on('click', async function () {
          if (explanation_id > 0) {
              explanation_id -= 1;
              select('#instance-counter')
                  .text("Instance " + explanation_id);
              update_explantion();
          }
      });

      // next button
      d3.select('button#next').on('click', function () {
          if (explanation_id < explanation_paths.length - 1) {
              explanation_id += 1;
              select('#instance-counter')
                  .text("Instance " + explanation_id);
              update_explantion();
          }
      });

      // save all button
      d3.select('button#save-all').on('click', function () {
          // store the current display + type
          var prev_style = current_display;
          var prev_type = current_expl_type;

          // save each style and display to an SVG 
          var viz_styles = [DisplayTypes.Numeric, DisplayTypes.Linguistic, DisplayTypes.Visual];
          var expl_types = [ExplanationTypes.Example, ExplanationTypes.Region];

          for (var i = 0; i < viz_styles.length; i++) {
              var style = viz_styles[i];
              current_display = style;
              update_display(style);
              for (var j = 0; j < expl_types.length; j++) {
                  var type = expl_types[j];
                  current_expl_type = type;
                  update_expl_type(type);
                  execute_save();
              }
          }

          // restore the display + type
          current_display = prev_style;
          current_expl_type = prev_type;
      });

  };

  main();

})();
//# sourceMappingURL=bundle.js.map
