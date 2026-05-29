(function () {
  function setExpanded(caption, list, expanded) {
    caption.setAttribute("aria-expanded", expanded ? "true" : "false");
    caption.classList.toggle("acme-expanded", expanded);
    list.classList.toggle("acme-expanded", expanded);
    list.style.display = expanded ? "block" : "none";
  }

  function initSidebarAccordion() {
    var tocRoot = document.querySelector("#bd-docs-nav .bd-toc-item");

    if (!tocRoot || tocRoot.dataset.acmeAccordionInit === "true") {
      return;
    }

    tocRoot.dataset.acmeAccordionInit = "true";

    var children = Array.prototype.slice.call(tocRoot.children);

    children.forEach(function (child, index) {
      if (!child.classList.contains("caption")) {
        return;
      }

      var list = child.nextElementSibling;
      if (!list || !list.classList.contains("bd-sidenav")) {
        return;
      }

      var listId = "acme-sidebar-group-" + index;
      var shouldExpand = !!list.querySelector(
        ".current, a[aria-current='page']"
      );

      child.classList.add("acme-caption-toggle");
      child.setAttribute("role", "button");
      child.setAttribute("tabindex", "0");
      child.setAttribute("aria-controls", listId);
      list.id = listId;
      list.classList.add("acme-collapsible-group");

      setExpanded(child, list, shouldExpand);

      var toggle = function () {
        var isExpanded = child.getAttribute("aria-expanded") === "true";
        setExpanded(child, list, !isExpanded);
      };

      child.addEventListener("click", toggle);
      child.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggle();
        }
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSidebarAccordion);
  } else {
    initSidebarAccordion();
  }
})();
